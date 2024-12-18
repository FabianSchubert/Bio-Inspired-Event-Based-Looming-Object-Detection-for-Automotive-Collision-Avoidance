from pygenn import genn_model

import numpy as np

'''
p_neuron = genn_model.create_custom_neuron_class(
    "P",
    param_names=["tau_m", "tau_i", "th"],
    derived_params=[
        (
            "alpha_m",
            genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))(),
        ),
        (
            "alpha_i",
            genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))(),
        )
    ],
    var_name_types=[("V", "scalar"), ("I", "scalar")],
    sim_code="""
    $(I) = $(alpha_i) * $(I) + (1.0 - $(alpha_i)) * $(Isyn);
    $(V) = $(alpha_m) * $(V) + (1.0 - $(alpha_m)) * $(I);
    """,
    threshold_condition_code="""
    $(V) > $(th)
    """,
    reset_code="$(V) = 0.0;",
    is_auto_refractory_required=False,
)
'''
p_neuron = genn_model.create_custom_neuron_class(
    "P",
    param_names=[],
    var_name_types=[("V", "scalar")],
    sim_code="$(V) = $(Isyn);",
    threshold_condition_code="$(V) > 0.0",
    reset_code=None,
    is_auto_refractory_required=False,
)

n_neuron = genn_model.create_custom_neuron_class(
    "N",
    param_names=[],
    var_name_types=[("V", "scalar")],
    sim_code="$(V) = $(Isyn);",
    threshold_condition_code="$(V) < 0.0",
    reset_code=None,
    is_auto_refractory_required=False,
)


s_neuron = genn_model.create_custom_neuron_class(
    "S",
    param_names=["v_reg"],
    var_name_types=[
        ("x_p", "scalar"),
        ("y_p", "scalar"),
        ("norm_p", "scalar"),
        ("one_to_one_p", "scalar"),
        ("x_n", "scalar"),
        ("y_n", "scalar"),
        ("norm_n", "scalar"),
        ("one_to_one_n", "scalar"),
        ("vx", "scalar"),
        ("vy", "scalar"),
        ("v_proj", "scalar"),
    ],
    sim_code="""
    $(one_to_one_p) = $(Isyn_p_one_to_one);
    $(one_to_one_n) = $(Isyn_n_one_to_one);

    const scalar vx_est_p = -$(one_to_one_p) * $(x_p) * DT / ($(v_reg) + DT * DT * $(norm_p));
    const scalar vy_est_p = -$(one_to_one_p) * $(y_p) * DT / ($(v_reg) + DT * DT * $(norm_p));

    const scalar vx_est_n = -$(one_to_one_n) * $(x_n) * DT / ($(v_reg) + DT * DT * $(norm_n));
    const scalar vy_est_n = -$(one_to_one_n) * $(y_n) * DT / ($(v_reg) + DT * DT * $(norm_n));

    $(vx) = vx_est_p + vx_est_n;
    $(vy) = vy_est_p + vy_est_n;

    $(x_p) = $(Isyn_p_x);
    $(y_p) = $(Isyn_p_y);
    $(norm_p) = $(Isyn_p_norm);

    $(x_n) = $(Isyn_n_x);
    $(y_n) = $(Isyn_n_y);
    $(norm_n) = $(Isyn_n_norm);

    $(v_proj) = $(vx) * $(x)[$(id)] + $(vy) * $(y)[$(id)];
    """,
    extra_global_params=[("x", "scalar*"), ("y", "scalar*")],
    additional_input_vars=[
        ("Isyn_p_x", "scalar", 0.0),
        ("Isyn_p_y", "scalar", 0.0),
        ("Isyn_p_norm", "scalar", 0.0),
        ("Isyn_p_one_to_one", "scalar", 0.0),
        ("Isyn_n_x", "scalar", 0.0),
        ("Isyn_n_y", "scalar", 0.0),
        ("Isyn_n_norm", "scalar", 0.0),
        ("Isyn_n_one_to_one", "scalar", 0.0),
    ],
    is_auto_refractory_required=False,
)

out_neuron = genn_model.create_custom_neuron_class(
    "OUT",
    param_names=[
        "output_scale",
        "tau_m",
        "filt_scale",
        "filt_bias",
    ],
    derived_params=[
        (
            "alpha_m",
            genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[2]))(),
        ),
    ],
    var_name_types=[
        ("r_right", "scalar"),
        ("r_left", "scalar"),
        ("V", "scalar"),
        ("V_linear", "scalar"),
    ],
    sim_code="""
    $(r_left) = $(Isyn_v_proj_left);
    $(r_right) = $(Isyn_v_proj_right);

    const scalar filt_l = 1.0 / (1.0 + exp(-($(r_left) * $(filt_scale) - $(filt_bias))));
    const scalar filt_r = 1.0 / (1.0 + exp(-($(r_right) * $(filt_scale) - $(filt_bias))));

    const scalar r_filt = $(output_scale) * min($(r_left), $(r_right)) * filt_l * filt_r;
    
    $(V) = $(alpha_m) * $(V) + (1.0 - $(alpha_m)) * r_filt;
    $(V_linear) = $(alpha_m) * $(V_linear) + (1.0 - $(alpha_m)) * ($(r_left) + $(r_right));
    """,
    additional_input_vars=[
        ("Isyn_v_proj_left", "scalar", 0.0),
        ("Isyn_v_proj_right", "scalar", 0.0),
    ],
)

sp_wu_with_v_pre = genn_model.create_custom_weight_update_class(
    "sp_wu_with_pre",
    param_names=[],
    var_name_types=[("g", "scalar")],
    sim_code="$(addToInSyn, $(g) * $(V_pre));",
)


def create_cont_wu(name: str, var_name: str):
    return genn_model.create_custom_weight_update_class(
        name,
        param_names=[],
        var_name_types=[("g", "scalar")],
        synapse_dynamics_code=f"$(addToInSyn, $(g) * $({var_name}_pre));",
    )


cont_wu = genn_model.create_custom_weight_update_class(
    "cont_wu",
    param_names=[],
    var_name_types=[("g", "scalar")],
    synapse_dynamics_code="$(addToInSyn, $(g) * $(V_pre));",
)

sparse_one_to_one_snippet_with_pad = genn_model.create_custom_sparse_connect_init_snippet_class(
    "sparse_one_to_one_snippet_with_pad",
    param_names=["pad_x", "pad_y", "width_pre", "height_pre"],
    row_build_code="""
    const int width_pre = (int)($(width_pre));
    const int height_pre = (int)($(height_pre));
    const int pad_x = (int)($(pad_x));
    const int pad_y = (int)($(pad_y));
    const int width_post = width_pre - 2 * pad_x;
    const int height_post = height_pre - 2 * pad_y;

    const int x_pre = $(id_pre) % width_pre;
    const int y_pre = $(id_pre) / width_pre;

    const int x_post = x_pre - pad_x;
    const int y_post = y_pre - pad_y;

    if(x_post >= 0 && x_post < width_post && y_post >= 0 && y_post < height_post) 
    {
        const int id_post = x_post + y_post * width_post;
        $(addSynapse, id_post);
    }
    $(endRow);
    """,
    calc_max_row_len_func=genn_model.create_cmlf_class(
        lambda num_pre, num_post, pars: 1
    )(),
    calc_max_col_len_func=genn_model.create_cmlf_class(
        lambda num_pre, num_post, pars: 1
    )(),
)
