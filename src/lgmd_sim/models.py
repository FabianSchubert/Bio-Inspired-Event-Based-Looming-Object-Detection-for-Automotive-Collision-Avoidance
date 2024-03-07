from pygenn import genn_model
import numpy as np

# curent source model for current input based on DVS events passed as a spike_array
spike_array_current_source = genn_model.create_custom_current_source_class(
    "",
    param_names=["unit_amplitude"],
    var_name_types=[("startSpike", "int"), ("endSpike", "int")],
    extra_global_params=[("spikeTimes", "scalar*"), ("spikePolarity", "int8_t*")],
    injection_code="""
    if ($(startSpike) != $(endSpike) && $(t) >= $(spikeTimes)[$(startSpike)]) {
        $(injectCurrent, $(unit_amplitude)*$(spikePolarity)[$(startSpike)]);
        $(startSpike)++;
    }
    if($(startSpike) > $(endSpike)){
        printf("Warning, startspike > endspike (should never happen)");
    }
    """,
)

bitmask_array_current_source = genn_model.create_custom_current_source_class(
    "",
    param_names=["unit_amplitude"],
    var_name_types=[("nt", "int"), ("pop_size", "int")],
    extra_global_params=[
        ("spikeBitmask", "uint32_t*"),
        ("polarityBitmask", "uint32_t*"),
    ],
    injection_code="""

    const unsigned int wordsPerTimestep = (($(pop_size) + 31) / 32);
    const uint32_t mySpikeWord = $(spikeBitmask)[(wordsPerTimestep * (int)round($(t) / DT)) + $(id) / 32];
    const uint32_t myPolarityWord = $(polarityBitmask)[(wordsPerTimestep * (int)round($(t) / DT)) + $(id) / 32];

    if(mySpikeWord & (1 << ($(id) % 32))){
        
        const int pol = (myPolarityWord & (1 << ($(id) % 32))) ? 1 : -1;
        $(injectCurrent,
            $(unit_amplitude)*(pol * 2.0 - 1.0)
        );
    }
    """,
)

# LIF neuron model for S neurons with soft reset as in Banchard et al. 2000
lif_neuron = genn_model.create_custom_neuron_class(
    "LIF",
    param_names=["tau_m", "V_thresh", "V_reset"],
    var_name_types=[("V", "scalar"), ("VI", "scalar")],
    sim_code="""
    $(VI) = $(Isyn);
    $(V) += ($(Isyn)-$(V))/$(tau_m)*DT;  // linear Euler
    """,
    threshold_condition_code="""
    $(V) >= $(V_thresh)
    """,
    reset_code="""
    $(V)-= $(V_reset);  // soft reset by $(V_reset)
    """,
    is_auto_refractory_required=False,
)

s_neuron = genn_model.create_custom_neuron_class(
    "S",
    param_names=["tau_m", "V_thresh", "V_reset"],
    var_name_types=[
        ("VI", "scalar"),
        ("VE", "scalar"),
        ("V", "scalar"),
        ("I", "scalar"),
    ],
    sim_code="""
    $(VI) = min(1.0, $(Isyn_I));
    $(VE) = $(Isyn_E);

    $(I) = $(VE) * (1.0 - $(VI));

    $(V) += ($(I) - $(V)) / $(tau_m) * DT;
    """,
    threshold_condition_code="$(V) >= $(V_thresh)",
    reset_code="""
    $(V)-= $(V_reset);  // soft reset by $(V_reset)
    """,
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
