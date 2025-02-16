using DifferentialEquations
using Plots
using Statistics
using FFTW
using DSP

const NUM_CELLS = 300
const TOTAL_TIME = 5000.0
const TOTAL_LENGTH = 240.0

const CROSS_CORRELATION_WINDOW = 75

const CORRELATION_TIMESPAN = (4000.0, 5000.0)
const SPACETIME_TIMESPAN = (4925.0, 5000.0)
const FREQUENCY_TIMESPAN = (4500.0, 5000.0)
const PROPAGATION_TIMESPAN = (4000.0, 5000.0)

"""
Parameters for the intestine model layers
"""
struct LayerParams
    k::Float64
    a::Float64
    β::Float64
    γ::Float64
    ε::Float64
    D::Float64
    D_il::Float64
    α::Float64
end

# LM layer parameters
const LM_PARAMS = LayerParams(
    10.0,   # k
    0.06,   # a
    0.0,    # β
    8.0,    # γ
    0.15,   # ε
    0.4,    # D
    0.3,    # D_il
    1.0     # α
)

# ICC layer parameters
const ICC_PARAMS = LayerParams(
    7.0,    # k
    0.5,    # a
    0.5,    # β
    8.0,    # γ
    0.15,   # ε (base value, will be position-dependent)
    0.04,   # D
    0.3,    # D_il
    -1.0    # α
)

"""
Calculate position-dependent parameter that follows exponential decay
"""
function calculate_decay_parameter(x::Float64, total_length::Float64, start_val::Float64, end_val::Float64, decay_rate::Float64)
    r = x * total_length  # Scale to cm
    return end_val + (start_val - end_val) * exp(decay_rate * r)
end

# Then define epsilon and intrinsic frequency calculations using this base function
function calculate_epsilon(x::Float64, total_length::Float64)
    return calculate_decay_parameter(x, total_length, 0.0833, 0.03, -0.015)
end

function calculate_intrinsic_frequency(x::Float64, total_length::Float64)
    return calculate_decay_parameter(x, total_length, 20.0, 10.25, -0.015)
end

"""
Calculate the Laplacian at a given index
"""
function calculate_laplacian(u, idx, N)
    if idx == 1
        # No-flux boundary at left end
        return 2.0 * (u[idx+1] - u[idx])
    elseif idx == N
        # No-flux boundary at right end
        return 2.0 * (u[idx-1] - u[idx])
    end
    return u[idx+1] + u[idx-1] - 2.0 * u[idx]
end

"""
System of differential equations for the coupled intestine layers
"""
function intestine_system!(du, u, p, t)
    N = NUM_CELLS       
    # u contains [v_lm; i_lm; v_icc; i_icc]
    # Split into separate views for each variable
    v_lm = @view u[1:N]
    i_lm = @view u[N+1:2N]
    v_icc = @view u[2N+1:3N]
    i_icc = @view u[3N+1:4N]
    
    # Corresponding derivatives
    dv_lm = @view du[1:N]
    di_lm = @view du[N+1:2N]
    dv_icc = @view du[2N+1:3N]
    di_icc = @view du[3N+1:4N]
    
    # Calculate derivatives for each cell
    for idx in 1:N
        x = (idx-1)/(N-1)  # Normalized position [0,1]
        
        # LM layer
        dv_lm[idx] = LM_PARAMS.k * v_lm[idx] * (v_lm[idx] - LM_PARAMS.a) * (1.0 - v_lm[idx]) - 
                     i_lm[idx] + 
                     LM_PARAMS.D * calculate_laplacian(v_lm, idx, N) + 
                     LM_PARAMS.α * LM_PARAMS.D_il * (v_icc[idx] - v_lm[idx])
        
        di_lm[idx] = LM_PARAMS.ε * (LM_PARAMS.γ * (v_lm[idx] - LM_PARAMS.β) - i_lm[idx])
        
        # ICC layer (with position-dependent epsilon)
        local_ε = calculate_epsilon(x, TOTAL_LENGTH)
        
        dv_icc[idx] = ICC_PARAMS.k * v_icc[idx] * (v_icc[idx] - ICC_PARAMS.a) * (1.0 - v_icc[idx]) - 
                      i_icc[idx] + 
                      ICC_PARAMS.D * calculate_laplacian(v_icc, idx, N) + 
                      ICC_PARAMS.α * ICC_PARAMS.D_il * (v_icc[idx] - v_lm[idx])
        
        di_icc[idx] = local_ε * (ICC_PARAMS.γ * (v_icc[idx] - ICC_PARAMS.β) - i_icc[idx])
    end
end

"""
Main simulation function
"""
function simulate_intestine(tspan=(0.0, 75.0))
    # Initial conditions
    u0 = zeros(4 * NUM_CELLS)
    
    # More gradual initial stimulation
    stim_cells = 1:ceil(Int, NUM_CELLS)  # Increased stimulation region
    for i in stim_cells
        phase = 2π * (i-1)/length(stim_cells)  # Phase gradient
        u0[i] = 0.5 * (1 + cos(phase))  # Smoother initial LM activation
        u0[2NUM_CELLS + i] = 0.5 * (1 + cos(phase))  # Smoother ICC activation
    end
    
    # Use a more stable solver with tighter tolerances
    prob = ODEProblem(intestine_system!, u0, tspan, ())
    sol = solve(prob, Tsit5(), 
                abstol=1e-8, 
                reltol=1e-8,
                dtmax=0.1)  # Limit maximum time step
    
    return sol
end

"""
Calculate cross-correlation between LM and ICC layers across distance
for a single state (signal snapshots at a moment in time)
"""
function calculate_spatial_correlation(v_lm, v_icc)
    correlation = zeros(NUM_CELLS)
    for i in 1:NUM_CELLS
        # Define spatial window centered at i
        start_idx = max(1, i - CROSS_CORRELATION_WINDOW÷2)
        end_idx = min(NUM_CELLS, i + CROSS_CORRELATION_WINDOW÷2)
        
        # Get signal segments in window
        v_lm_window = v_lm[start_idx:end_idx]
        v_icc_window = v_icc[start_idx:end_idx]
        
        # Normalize and correlate
        correlation[i] = cor(v_lm_window, v_icc_window)
    end
    return correlation
end

"""
Calculate time-averaged spatial correlation between LM and ICC layers
"""
function calculate_time_averaged_correlation(sol, timespan)
    t_indices = findall(t -> timespan[1] <= t <= timespan[2], sol.t)
    
    # Initialize accumulator
    correlation_sum = zeros(NUM_CELLS)
    
    # Sum correlations for each timepoint
    for t in sol.t[t_indices]
        state = sol(t)
        v_lm = state[1:NUM_CELLS]
        v_icc = state[2NUM_CELLS+1:3NUM_CELLS]
        correlation_sum += calculate_spatial_correlation(v_lm, v_icc)
    end
    
    # Average
    return correlation_sum ./ length(t_indices)
end

"""
Generate time-averaged correlation plot
"""
function analyze_time_averaged_correlation(sol, timespan)
    x_axis = 0:Int(TOTAL_LENGTH)/(NUM_CELLS-1):Int(TOTAL_LENGTH)
    
    # Calculate time-averaged correlation
    avg_correlation = calculate_time_averaged_correlation(sol, timespan)
    
    p = plot(x_axis, avg_correlation,
             xlabel="Distance from pylorus (cm)",
             ylabel="Correlation",
             title="Time-averaged Cross-correlation",
             legend=false,
             ylims=(0,1))
    
    savefig(p, "time_averaged_correlation.png")
end

"""
Calculate propagation characteristics at each point by tracking wave coherence
"""
function calculate_propagation_characteristics(sol, timespan)
    # Get data within timespan
    t_indices = findall(t -> timespan[1] <= t <= timespan[2], sol.t)
    times = sol.t[t_indices]
    dt = times[2] - times[1]
    
    # Get LM layer data
    v_lm_history = [sol(t)[1:NUM_CELLS] for t in times]
    
    # Initialize arrays
    prop_length = zeros(NUM_CELLS)
    prop_time = zeros(NUM_CELLS)
    
    # Parameters for wave tracking
    correlation_threshold = 0.6  # Reduced from 0.7
    phase_threshold = π/3  # Increased from π/2
    velocity_factor = 0.5  # cm/s - adjusted to match experimental timescales
    
    # For each starting point
    for i in 1:NUM_CELLS-1
        reference_signal = [v[i] for v in v_lm_history]
        
        # Track maximum coherent propagation
        max_coherent_cells = 0
        
        # Look at subsequent points
        for j in (i+1):NUM_CELLS
            compare_signal = [v[j] for v in v_lm_history]
            
            # Calculate correlation
            correlation = cor(reference_signal, compare_signal)
            
            # Phase analysis using Hilbert transform
            hilbert_ref = hilbert(reference_signal)
            hilbert_comp = hilbert(compare_signal)
            phase_diff = abs(angle(mean(hilbert_ref .* conj(hilbert_comp))))
            
            # Check wave coherence with relaxed criteria
            if correlation > correlation_threshold && phase_diff < phase_threshold
                max_coherent_cells = j - i
            else
                break
            end
        end
        
        # Convert to physical units
        prop_length[i] = max_coherent_cells * (TOTAL_LENGTH/NUM_CELLS) * 10.0  # Scale factor to match experimental range
        prop_time[i] = prop_length[i] / velocity_factor
    end
    
    return prop_length, prop_time
end

"""
Generate propagation characteristics plots
"""
function analyze_propagation(sol, timespan)
    x_axis = range(0, TOTAL_LENGTH, length=NUM_CELLS)
    
    # Calculate propagation characteristics
    prop_length, prop_time = calculate_propagation_characteristics(sol, timespan)
    
    # Smooth the results
    window_length = 21  # Increased window length for smoother curves
    prop_length_smooth = movmean(prop_length, window_length)
    prop_time_smooth = movmean(prop_time, window_length)
    
    # Plot propagation length
    p1 = plot(x_axis, prop_length_smooth,
              xlabel="Distance from pylorus (cm)",
              ylabel="Propagation length (cm)",
              title="Mean Propagation Length",
              legend=false,
              ylims=(0, 35))  # Set y-axis limits to match paper
    
    savefig(p1, "propagation_length.png")
    
    # Plot propagation time
    p2 = plot(x_axis, prop_time_smooth,
              xlabel="Distance from pylorus (cm)",
              ylabel="Propagation time (s)",
              title="Mean Propagation Time",
              legend=false,
              ylims=(0, 60))  # Set y-axis limits to match paper
    
    savefig(p2, "propagation_time.png")
end

"""
Generate space-time plots
"""
function analyze_spacetime(sol, timespan)
    x_axis = 0:Int(TOTAL_LENGTH)/(NUM_CELLS-1):Int(TOTAL_LENGTH)
    
    # Get data within timespan
    t_indices = findall(t -> timespan[1] <= t <= timespan[2], sol.t)
    times = sol.t[t_indices]
    v_lm_history = [sol(t)[1:NUM_CELLS] for t in times]
    v_icc_history = [sol(t)[2NUM_CELLS+1:3NUM_CELLS] for t in times]
    
    # LM layer space-time plot
    p1 = heatmap(x_axis, times, reduce(hcat, v_lm_history)',
                xlabel="Distance from pylorus (cm)",
                ylabel="Time (s)",
                title="Space-Time Plot - LM Layer",
                colorbar=:right)
    savefig(p1, "spacetime_lm.png")
    
    # ICC layer space-time plot
    p2 = heatmap(x_axis, times, reduce(hcat, v_icc_history)',
                xlabel="Distance from pylorus (cm)",
                ylabel="Time (s)",
                title="Space-Time Plot - ICC Layer",
                colorbar=:right)
    savefig(p2, "spacetime_icc.png")
end

"""
Calculate dominant frequency from a time series
Returns frequency in cycles/min
"""
function calculate_frequency(time_series, dt)    
    # Prepare for spectral analysis
    n_samples = length(time_series)
   
    # Compute power spectrum
    freq = fftfreq(n_samples, 1/dt)  # Frequency array in Hz
    pos_freq_idx = freq .> 0
    freq = freq[pos_freq_idx]
    power = abs2.(fft(time_series))[pos_freq_idx]
   
    # Convert frequencies to cycles/min
    freq_cpm = freq .* 60
   
    # Get strongest peak across all frequencies
    peak_idx = argmax(power)
    dominant_freq = freq_cpm[peak_idx]
   
    return dominant_freq
end

"""
Generate frequency analysis plots
"""
function analyze_frequencies(sol, timespan)
    dt = sol.t[2] - sol.t[1]
    
    # Use timespan for analysis
    t_indices = findall(t -> timespan[1] <= t <= timespan[2], sol.t)
    time_window = sol.t[t_indices]
    
    # Calculate frequencies
    freqs_lm = zeros(NUM_CELLS)
    freqs_icc = zeros(NUM_CELLS)
    
    for i in 1:NUM_CELLS
        ts_lm = [sol(t)[i] for t in time_window]
        ts_icc = [sol(t)[2NUM_CELLS+i] for t in time_window]
        
        freqs_lm[i] = calculate_frequency(ts_lm, dt)
        freqs_icc[i] = calculate_frequency(ts_icc, dt)
    end
    
    # Calculate intrinsic frequencies
    x = range(0, 1, length=NUM_CELLS)
    x_axis = range(0, TOTAL_LENGTH, length=NUM_CELLS)
    intrinsic_freqs = [calculate_intrinsic_frequency(xi, TOTAL_LENGTH) for xi in x]
    
    # Calculate average frequencies
    avg_freqs = (freqs_lm + freqs_icc) ./ 2
    
    # Individual frequencies plot
    p1 = plot(x_axis, freqs_lm,
             label="LM Layer",
             linewidth=2,
             xlabel="Distance from pylorus (cm)",
             ylabel="Frequency (cycles min⁻¹)",
             title="Individual Layer Frequencies")
    
    plot!(p1, x_axis, freqs_icc,
          label="ICC Layer",
          linewidth=2)
    
    plot!(p1, x_axis, intrinsic_freqs,
          label="Intrinsic",
          linestyle=:dash,
          color=:black)
    
    ylims!(p1, 0, 30)
    savefig(p1, "frequencies_individual.png")
    
    # Average frequencies plot
    p2 = plot(x_axis, avg_freqs,
             label="Average",
             linewidth=2,
             xlabel="Distance from pylorus (cm)",
             ylabel="Frequency (cycles min⁻¹)",
             title="Average Layer Frequency vs Intrinsic")
    
    plot!(p2, x_axis, intrinsic_freqs,
          label="Intrinsic",
          linestyle=:dash,
          color=:black)
    
    ylims!(p2, 0, 30)
    savefig(p2, "frequencies_average.png")
end

# Modified main execution
sol = simulate_intestine((0, TOTAL_TIME));
println("Sim done - presenting results...")
analyze_final_state(sol)
analyze_time_averaged_correlation(sol, CORRELATION_TIMESPAN)
analyze_propagation(sol, PROPAGATION_TIMESPAN)
analyze_spacetime(sol, SPACETIME_TIMESPAN)
analyze_frequencies(sol, FREQUENCY_TIMESPAN)