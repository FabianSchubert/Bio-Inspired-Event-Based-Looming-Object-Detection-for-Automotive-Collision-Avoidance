from pygenn import genn_model

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

p_neuron = genn_model.create_custom_neuron_class(
    "P",
    param_names=["tau_m", "tau_i", "V_thresh", "V_reset"],
    var_name_types=[("V", "scalar"), ("VI", "scalar")],
    sim_code="""
    $(V) += DT * ($(VI) - $(V)) / $(tau_m);
    $(VI) += DT * ($(Isyn) - $(VI)) / $(tau_i);
    """,
    threshold_condition_code="""
    $(V) >= $(V_thresh)
    """,
    reset_code="""
    $(V) = $(V_reset);  // reset to $(V_reset)
    """,
    is_auto_refractory_required=False,
)

# LIF neuron model
lif_neuron = genn_model.create_custom_neuron_class(
    "LIF",
    param_names=["tau_m", "V_thresh", "V_reset"],
    var_name_types=[("V", "scalar"), ("VI", "scalar")],
    sim_code="""
    $(VI) = $(Isyn);
    $(V) += DT * ($(VI)-$(V))/$(tau_m);  // linear Euler
    """,
    threshold_condition_code="""
    $(V) >= $(V_thresh)
    """,
    reset_code="""
    $(V) = $(V_reset);  // reset to $(V_reset)
    """,
    is_auto_refractory_required=False,
)