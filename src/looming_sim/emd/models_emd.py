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
    threshold_condition_code="$(V) != 0.0",
    reset_code=None,
    is_auto_refractory_required=False,
)

s_neuron = genn_model.create_custom_neuron_class(
    "S",
    param_names=["tau_v", "tau_px", "v_reg"],
    var_name_types=[
        ("px", "scalar"),
        ("py", "scalar"),
        ("px_prev", "scalar"),
        ("py_prev", "scalar"),
        ("sp_in", "scalar"),
        ("sp_in_prev", "scalar"),
        ("vx", "scalar"),
        ("vy", "scalar"),
        ("v_proj", "scalar"),
        ("act_filt", "scalar"),
    ],
    derived_params=[
        (
            "alpha_v",
            genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))(),
        ),
        (
            "alpha_px",
            genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))(),
        )
    ],
    sim_code="""
    $(px_prev) = $(px);
    $(py_prev) = $(py);
    $(sp_in_prev) = $(sp_in);

    $(px) = $(px) * $(alpha_px) + (1.-$(alpha_px)) * $(Isyn_x);
    $(py) = $(py) * $(alpha_px) + (1.-$(alpha_px)) * $(Isyn_y);
    $(sp_in) = $(sp_in) * $(alpha_px) + (1.-$(alpha_px)) * $(Isyn_sp_in);

    $(act_filt) = $(alpha_v) * $(act_filt) + (1.0 - $(alpha_v)) * $(sp_in);

    const scalar vx_est = ($(px) * $(sp_in_prev) - $(px_prev) * $(sp_in)) / (DT * ($(v_reg) + $(sp_in) * $(sp_in_prev)));
    const scalar vy_est = ($(py) * $(sp_in_prev) - $(py_prev) * $(sp_in)) / (DT * ($(v_reg) + $(sp_in) * $(sp_in_prev)));

    $(vx) = $(alpha_v) * $(vx) + (1.0 - $(alpha_v)) * vx_est;
    $(vy) = $(alpha_v) * $(vy) + (1.0 - $(alpha_v)) * vy_est;

    $(v_proj) = $(vx) * $(x)[$(id)] + $(vy) * $(y)[$(id)];
    """,
    extra_global_params=[("x", "scalar*"), ("y", "scalar*")],
    additional_input_vars=[
        ("Isyn_x", "scalar", 0.0),
        ("Isyn_y", "scalar", 0.0),
        ("Isyn_sp_in", "scalar", 0.0),
    ],
    is_auto_refractory_required=False,
)

out_neuron = genn_model.create_custom_neuron_class(
    "OUT",
    param_names=[
        "output_scale",
        "r_reg",
        "tau_m",
        "tau_r",
        "filt_scale",
        "filt_bias",
        "pos_norm_mean_left",
        "pos_norm_mean_right",
    ],
    derived_params=[
        (
            "alpha_m",
            genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[2]))(),
        ),
        (
            "alpha_r",
            genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[3]))(),
        ),
    ],
    var_name_types=[("r_right", "scalar"), ("r_left", "scalar"), ("V", "scalar")],
    sim_code="""
    const scalar r_l_ = $(Isyn_v_proj_left) / ($(r_reg) + $(pos_norm_mean_left));// * $(Isyn_avg_act_left));
    const scalar r_r_ = $(Isyn_v_proj_right) / ($(r_reg) + $(pos_norm_mean_right));// * $(Isyn_avg_act_right));

    $(r_left) = $(alpha_r) * $(r_left) + (1.0 - $(alpha_r)) * r_l_ * $(output_scale);
    $(r_right) = $(alpha_r) * $(r_right) + (1.0 - $(alpha_r)) * r_r_ * $(output_scale);

    const scalar filt_l = 1.0 / (1.0 + exp(-($(r_left) - $(filt_bias)) / $(filt_scale)));
    const scalar filt_r = 1.0 / (1.0 + exp(-($(r_right) - $(filt_bias)) / $(filt_scale)));

    const scalar r_filt = 0.5 * ($(r_left) + $(r_right)) * filt_l * filt_r;
    
    $(V) = $(alpha_m) * $(V) + (1.0 - $(alpha_m)) * r_filt;
    """,
    additional_input_vars=[
        ("Isyn_v_proj_left", "scalar", 0.0),
        ("Isyn_v_proj_right", "scalar", 0.0),
        ("Isyn_avg_act_left", "scalar", 0.0),
        ("Isyn_avg_act_right", "scalar", 0.0),
    ],
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
