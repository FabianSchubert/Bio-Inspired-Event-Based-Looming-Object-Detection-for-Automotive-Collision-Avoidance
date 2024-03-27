from pygenn import genn_model
import numpy as np

lgmd_neuron = genn_model.create_custom_neuron_class(
    "LGMD",
    param_names=["tau_m", "V_thresh", "V_reset", "scale_i_in"],
    var_name_types=[("V", "scalar"), ("VI", "scalar")],
    sim_code="""
    $(VI) = $(Isyn) * exp($(Isyn_i)/$(scale_i_in));
    $(V) += ($(VI)-$(V))/$(tau_m)*DT;  // linear Euler
    """,
    threshold_condition_code="""
    $(V) >= $(V_thresh)
    """,
    reset_code="""
    $(V)-= $(V_reset);  // soft reset by $(V_reset)
    """,
    additional_input_vars=[("Isyn_i", "scalar", 0.0)],
    is_auto_refractory_required=False,
)

# connectivity initialisation snippet for 1:1 connectivity between input and S
one_one_with_boundary = genn_model.create_custom_sparse_connect_init_snippet_class(
    "one_one_with_boundary",
    # assume out < in and out centered on in
    param_names=["in_ht", "in_wd", "out_yoff", "out_xoff"],
    row_build_code="""
        const int in_x= $(id_pre)%((int) $(in_wd));
        const int in_y= $(id_pre)/((int) $(in_wd));
        const int out_wd= ((int) $(in_wd))-2*((int) $(out_xoff)); 
        const int out_ht= ((int) $(in_ht))-2*((int) $(out_yoff)); 
        const int out_x= in_x-((int) $(out_xoff));
        if ((out_x >= 0) && (out_x < out_wd)) {
            const int out_y= in_y-((int) $(out_yoff));
            if ((out_y >= 0) && (out_y < out_ht)) {
                $(addSynapse, (out_y*out_wd+out_x));
            }
        }
        $(endRow);
        """,
    calc_max_row_len_func=genn_model.create_cmlf_class(
        lambda num_pre, num_post, pars: 1
    )(),
)

# postsynaptic model to emulate the thresholded linear unit F action on LGMD
threshold_exp_curr = genn_model.create_custom_postsynaptic_class(
    "threshold_exp_curr",
    param_names=["tau", "threshold"],
    derived_params=[
        (
            "expDecay",
            genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))(),
        ),
        (
            "init",
            genn_model.create_dpf_class(
                lambda pars, dt: (pars[0] * (1.0 - np.exp(-dt / pars[0]))) * (1.0 / dt)
            )(),
        ),
    ],
    decay_code="$(inSyn)*= $(expDecay);",
    apply_input_code="if ($(init)*$(inSyn) < -$(threshold)) $(Isyn)+= $(init)*$(inSyn);",
)
