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
