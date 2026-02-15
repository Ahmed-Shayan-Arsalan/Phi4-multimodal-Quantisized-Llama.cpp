#include "models.h"

ggml_cgraph * clip_graph_conformer::build() {
    const int n_frames   = img.nx;
    const int n_pos      = n_frames / 2;
    const int n_pos_embd = (((((n_frames + 1) / 2) + 1) / 2 + 1) / 2) * 2 - 1;
    GGML_ASSERT(model.position_embeddings->ne[1] >= n_pos);

    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, n_pos_embd);
    ggml_set_name(pos_emb, "pos_emb");
    ggml_set_input(pos_emb);
    ggml_build_forward_expand(gf, pos_emb);

    ggml_tensor * inp = build_inp_raw(1);

    auto * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    // Global mel normalization: (mel - global_mean) * global_invstd
    // This normalizes raw mel spectrogram features to zero-mean, unit-variance
    // using pretrained statistics from the audio encoder
    if (model.global_mean && model.global_invstd) {
        cur = ggml_sub(ctx0, cur, model.global_mean);
        cur = ggml_mul(ctx0, cur, model.global_invstd);
        cb(cur, "conformer.global_norm", -1);
    }

    // pre encode, conv subsampling
    {
        // layer.0 - conv2d
        cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[0], cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[0]);
        cb(cur, "conformer.pre_encode.conv.{}", 0);

        // layer.1 - relu
        cur = ggml_relu_inplace(ctx0, cur);

        // layer.2 conv2d dw
        cur = ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[2], cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[2]);
        cb(cur, "conformer.pre_encode.conv.{}", 2);

        // layer.3 conv2d
        cur = ggml_conv_2d_direct(ctx0, model.pre_encode_conv_X_w[3], cur, 1, 1, 0, 0, 1, 1);
        cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[3]);
        cb(cur, "conformer.pre_encode.conv.{}", 3);

        // layer.4 - relu
        cur = ggml_relu_inplace(ctx0, cur);

        // layer.5 conv2d dw
        cur = ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[5], cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[5]);
        cb(cur, "conformer.pre_encode.conv.{}", 5);

        // layer.6 conv2d
        cur = ggml_conv_2d_direct(ctx0, model.pre_encode_conv_X_w[6], cur, 1, 1, 0, 0, 1, 1);
        cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[6]);
        cb(cur, "conformer.pre_encode.conv.{}", 6);

        // layer.7 - relu
        cur = ggml_relu_inplace(ctx0, cur);

        // flatten channel and frequency axis
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);

        // calculate out
        cur = ggml_mul_mat(ctx0, model.pre_encode_out_w, cur);
        cur = ggml_add(ctx0, cur, model.pre_encode_out_b);
        cb(cur, "conformer.pre_encode.out", -1);
    }

    // pos_emb
    cb(pos_emb, "pos_emb", -1);

    // T5 relative attention bias (if available, use instead of NeMo-style pos bias)
    const bool use_t5_bias = (model.rel_attn_bias != nullptr);
    ggml_tensor * t5_bias = nullptr;
    if (use_t5_bias) {
        const int T = (int)cur->ne[1];
        t5_bias = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, T, T, n_head);
        ggml_set_name(t5_bias, "t5_bias");
        ggml_set_input(t5_bias);
        ggml_build_forward_expand(gf, t5_bias);
    }

    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];

        auto * residual = cur;

        cb(cur, "layer.in", il);

        // feed_forward1
        cur = build_norm(cur, layer.ff_norm_w, layer.ff_norm_b, NORM_TYPE_NORMAL, 1e-5, il);
        cb(cur, "conformer.layers.{}.norm_feed_forward1", il);

        cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w, layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, FFN_SILU,
                        il);
        cb(cur, "conformer.layers.{}.feed_forward1.linear2", il);

        const auto fc_factor = 0.5f;
        residual             = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, fc_factor));

        // self-attention
        {
            cur = build_norm(residual, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, 1e-5, il);
            cb(cur, "conformer.layers.{}.norm_self_att", il);

            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
            Qcur               = ggml_add(ctx0, Qcur, layer.q_b);
            Qcur               = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, Qcur->ne[1]);

            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
            Kcur               = ggml_add(ctx0, Kcur, layer.k_b);
            Kcur               = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, Kcur->ne[1]);
            Kcur               = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
            Vcur               = ggml_add(ctx0, Vcur, layer.v_b);
            Vcur               = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, Vcur->ne[1]);
            Vcur               = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3));

            ggml_tensor * scores;

            if (use_t5_bias) {
                // T5-style attention: simple Q*K^T + precomputed relative bias
                auto * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);  // (d_head, T, n_head)

                scores = ggml_mul_mat(ctx0, Q, Kcur);  // (T, T, n_head)
                scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));
                scores = ggml_scale(ctx0, scores, 1.0f / std::sqrt((float)d_head));
                scores = ggml_add(ctx0, scores, t5_bias);
            } else {
                // NeMo-style attention with pos_bias_u/v and relative shift
                ggml_tensor * Q_bias_u = ggml_add(ctx0, Qcur, layer.pos_bias_u);
                Q_bias_u               = ggml_permute(ctx0, Q_bias_u, 0, 2, 1, 3);
                ggml_tensor * Q_bias_v = ggml_add(ctx0, Qcur, layer.pos_bias_v);
                Q_bias_v               = ggml_permute(ctx0, Q_bias_v, 0, 2, 1, 3);

                ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, Q_bias_u, Kcur);
                matrix_ac               = ggml_cont(ctx0, ggml_permute(ctx0, matrix_ac, 1, 0, 2, 3));
                cb(matrix_ac, "conformer.layers.{}.self_attn.id3", il);

                auto * p = ggml_mul_mat(ctx0, layer.linear_pos_w, pos_emb);
                cb(p, "conformer.layers.{}.self_attn.linear_pos", il);
                p = ggml_reshape_3d(ctx0, p, d_head, n_head, p->ne[1]);
                p = ggml_permute(ctx0, p, 0, 2, 1, 3);

                auto * matrix_bd = ggml_mul_mat(ctx0, Q_bias_v, p);
                matrix_bd        = ggml_cont(ctx0, ggml_permute(ctx0, matrix_bd, 1, 0, 2, 3));

                // rel shift
                {
                    const auto pos_len = matrix_bd->ne[0];
                    const auto q_len   = matrix_bd->ne[1];
                    const auto h       = matrix_bd->ne[2];
                    matrix_bd          = ggml_pad(ctx0, matrix_bd, 1, 0, 0, 0);
                    matrix_bd          = ggml_roll(ctx0, matrix_bd, 1, 0, 0, 0);
                    matrix_bd          = ggml_reshape_3d(ctx0, matrix_bd, q_len, pos_len + 1, h);
                    matrix_bd          = ggml_view_3d(ctx0, matrix_bd, q_len, pos_len, h, matrix_bd->nb[1],
                                                            matrix_bd->nb[2], matrix_bd->nb[0] * q_len);
                    matrix_bd          = ggml_cont_3d(ctx0, matrix_bd, pos_len, q_len, h);
                }

                matrix_bd = ggml_view_3d(ctx0, matrix_bd, matrix_ac->ne[0], matrix_bd->ne[1],
                                                   matrix_bd->ne[2], matrix_bd->nb[1], matrix_bd->nb[2], 0);
                scores    = ggml_add(ctx0, matrix_ac, matrix_bd);
                scores    = ggml_scale(ctx0, scores, 1.0f / std::sqrt((float)d_head));
            }
            cb(scores, "conformer.layers.{}.self_attn.id0", il);

            ggml_tensor * attn = ggml_soft_max(ctx0, scores);
            ggml_tensor * x    = ggml_mul_mat(ctx0, attn, Vcur);
            x                  = ggml_permute(ctx0, x, 2, 0, 1, 3);
            x                  = ggml_cont_2d(ctx0, x, x->ne[0] * x->ne[1], x->ne[2]);

            ggml_tensor * out = ggml_mul_mat(ctx0, layer.o_w, x);
            out               = ggml_add(ctx0, out, layer.o_b);
            cb(out, "conformer.layers.{}.self_attn.linear_out", il);

            cur = out;
        }

        residual = ggml_add(ctx0, residual, cur);
        cur      = build_norm(residual, layer.norm_conv_w, layer.norm_conv_b, NORM_TYPE_NORMAL, 1e-5, il);
        cb(cur, "conformer.layers.{}.norm_conv", il);

        // conv
        {
            auto * x = cur;
            x = ggml_mul_mat(ctx0, layer.conv_pw1_w, x);
            x = ggml_add(ctx0, x, layer.conv_pw1_b);
            cb(x, "conformer.layers.{}.conv.pointwise_conv1", il);

            // Phi-4 conv_glu_type = "swish": (value + b1) * silu(gate + b2)
            // where silu(x) = x * sigmoid(x) = swish(x)
            {
                int64_t d = x->ne[0] / 2;
                ggml_tensor * value = ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0);
                ggml_tensor * gate  = ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d * x->nb[0]);
                if (layer.conv_glu_b1) {
                    value = ggml_add(ctx0, value, layer.conv_glu_b1);
                }
                if (layer.conv_glu_b2) {
                    gate = ggml_add(ctx0, gate, layer.conv_glu_b2);
                }
                gate = ggml_silu(ctx0, gate);
                x    = ggml_mul(ctx0, value, gate);
                x    = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            }

            // use ggml_ssm_conv for f32 precision
            // padding = (kernel_size - 1) / 2 for "same" (symmetric) padding
            {
                const int d_conv = (int)layer.conv_dw_w->ne[0]; // kernel size
                const int pad    = (d_conv - 1) / 2;
                x = ggml_pad(ctx0, x, pad, 0, 0, 0);
                x = ggml_roll(ctx0, x, pad, 0, 0, 0);
                x = ggml_pad(ctx0, x, pad, 0, 0, 0);
            }
            x = ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            x = ggml_add(ctx0, x, layer.conv_dw_b);

            // post-depthwise pointwise conv (Phi-4's dw_sep_conv_1d.pw_conv)
            if (layer.conv_pw_mid_w) {
                x = ggml_mul_mat(ctx0, layer.conv_pw_mid_w, x);
                if (layer.conv_pw_mid_b) {
                    x = ggml_add(ctx0, x, layer.conv_pw_mid_b);
                }
            } else {
                // fallback: use conv_norm as element-wise scale+bias (for models without pw_mid)
                x = ggml_add(ctx0, ggml_mul(ctx0, x, layer.conv_norm_w), layer.conv_norm_b);
            }
            x = ggml_silu(ctx0, x);

            // pointwise_conv2
            x = ggml_mul_mat(ctx0, layer.conv_pw2_w, x);
            x = ggml_add(ctx0, x, layer.conv_pw2_b);

            cur = x;
        }

        residual = ggml_add(ctx0, residual, cur);

        cur = build_norm(residual, layer.ff_norm_1_w, layer.ff_norm_1_b, NORM_TYPE_NORMAL, 1e-5, il);
        cb(cur, "conformer.layers.{}.norm_feed_forward2", il);

        cur = build_ffn(cur, layer.ff_up_1_w, layer.ff_up_1_b, layer.ff_gate_1_w, layer.ff_gate_1_b, layer.ff_down_1_w, layer.ff_down_1_b,
                        FFN_SILU, il);  // Phi-4 uses value * silu(gate) = SwiGLU
        cb(cur, "conformer.layers.{}.feed_forward2.linear2", il);

        residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, fc_factor));
        cb(residual, "conformer.layers.{}.conv.id", il);

        cur = build_norm(residual, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, 1e-5, il);
        cb(cur, "conformer.layers.{}.norm_out", il);
    }

    // audio adapter
    cur = build_norm(cur, model.mm_0_w, model.mm_0_b, NORM_TYPE_NORMAL, 1e-5, -1);
    cb(cur, "audio_adapter.model.{}", 0);
    cur = build_ffn(cur, model.mm_1_w, model.mm_1_b, nullptr, nullptr, model.mm_3_w, model.mm_3_b, FFN_GELU_ERF, -1);

    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
