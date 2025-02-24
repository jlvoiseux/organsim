"""
Implementation of a simple nonlinear model of electrical activity in the intestine
https://jlvoiseux.com/blog/20250218-simulating-electrical-activity-in-the-intestine/
"""

using DifferentialEquations, Plots, Statistics, FFTW, DSP

"""
Global simulation constants
"""
const NUM_CELLS = 300
const TOTAL_TIME = 5000.0
const TOTAL_LENGTH = 240.0

const CROSS_CORRELATION_WINDOW = 75
const FREQ_MOVMEAN_WINDOW = 5

const CORRELATION_TIMESPAN = (4000.0, 5000.0)
const SPACETIME_TIMESPAN   = (4925.0, 5000.0)
const FREQUENCY_TIMESPAN   = (4000.0, 5000.0)

"""
Struct to hold parameters for an intestinal layer.
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

"""
LM layer parameters
"""
const LM_PARAMS = LayerParams(
    10.0, 0.06, 0.0, 8.0, 0.15, 0.4, 0.3, 1.0
)

"""
ICC layer parameters (with position‐dependent ε)
"""
const ICC_PARAMS = LayerParams(
    7.0, 0.5, 0.5, 8.0, 0.15, 0.04, 0.3, -1.0
)

"""
Calculate a parameter with exponential decay based on position.
x is normalized (0–1); total_length is in cm.
"""
function calculate_decay_parameter(x::Float64, total_length::Float64, start_val::Float64, end_val::Float64, decay_rate::Float64)
    r = x * total_length
    return end_val + (start_val - end_val) * exp(decay_rate * r)
end

calculate_epsilon(x, total_length) = calculate_decay_parameter(x, total_length, 0.0833, 0.03, -0.015)
calculate_intrinsic_frequency(x, total_length) = calculate_decay_parameter(x, total_length, 20.0, 10.25, -0.014)

"""
Compute the discrete Laplacian at index idx with no-flux boundaries.
"""
function calculate_laplacian(u, idx, N)
    if idx == 1
        return 2.0 * (u[idx+1] - u[idx])
    elseif idx == N
        return 2.0 * (u[idx-1] - u[idx])
    end
    return u[idx+1] + u[idx-1] - 2.0 * u[idx]
end

"""
System of ODEs for the coupled intestine layers.
State vector u = [v_lm; i_lm; v_icc; i_icc].
"""
function intestine_system!(du, u, p, t)
    N = NUM_CELLS       
    v_lm  = @view u[1:N]
    i_lm  = @view u[N+1:2N]
    v_icc = @view u[2N+1:3N]
    i_icc = @view u[3N+1:4N]
    
    dv_lm  = @view du[1:N]
    di_lm  = @view du[N+1:2N]
    dv_icc = @view du[2N+1:3N]
    di_icc = @view du[3N+1:4N]
    
    for idx in 1:N
        x = (idx - 1) / (N - 1)  # Normalized position [0,1]
        
        # LM layer dynamics
        dv_lm[idx] = LM_PARAMS.k * v_lm[idx] * (v_lm[idx] - LM_PARAMS.a) * (1.0 - v_lm[idx]) -
                     i_lm[idx] +
                     LM_PARAMS.D * calculate_laplacian(v_lm, idx, N) +
                     LM_PARAMS.α * LM_PARAMS.D_il * (v_icc[idx] - v_lm[idx])
        di_lm[idx] = LM_PARAMS.ε * (LM_PARAMS.γ * (v_lm[idx] - LM_PARAMS.β) - i_lm[idx])
        
        # ICC layer dynamics (with position-dependent ε)
        local_ε = calculate_epsilon(x, TOTAL_LENGTH)
        dv_icc[idx] = ICC_PARAMS.k * v_icc[idx] * (v_icc[idx] - ICC_PARAMS.a) * (1.0 - v_icc[idx]) -
                      i_icc[idx] +
                      ICC_PARAMS.D * calculate_laplacian(v_icc, idx, N) +
                      ICC_PARAMS.α * ICC_PARAMS.D_il * (v_icc[idx] - v_lm[idx])
        di_icc[idx] = local_ε * (ICC_PARAMS.γ * (v_icc[idx] - ICC_PARAMS.β) - i_icc[idx])
    end
end

"""
Simulate the intestine model over the given time span.
"""
function simulate_intestine(tspan=(0.0, 75.0))
    u0 = zeros(4 * NUM_CELLS)
    
    # Apply initial stimulation with a phase gradient to both layers.
    stim_cells = 1:ceil(Int, NUM_CELLS)
    for i in stim_cells
        phase = 2π * (i - 1) / length(stim_cells)
        u0[i] = 0.5 * (1 + cos(phase))
        u0[2 * NUM_CELLS + i] = 0.5 * (1 + cos(phase))
    end
    
    prob = ODEProblem(intestine_system!, u0, tspan, ())
    sol = solve(prob, Tsit5(); abstol=1e-8, reltol=1e-8, dtmax=0.05)
    return sol
end

"""
Calculate the spatial cross-correlation between two signals over a sliding window.
"""
function calculate_spatial_correlation(v_lm, v_icc)
    correlation = zeros(NUM_CELLS)
    for i in 1:NUM_CELLS
        start_idx = max(1, i - CROSS_CORRELATION_WINDOW ÷ 2)
        end_idx   = min(NUM_CELLS, i + CROSS_CORRELATION_WINDOW ÷ 2)
        correlation[i] = cor(v_lm[start_idx:end_idx], v_icc[start_idx:end_idx])
    end
    return correlation
end

"""
Generate final state plots: LM & ICC potentials and their cross-correlation.
"""
function analyze_final_state(sol)
    x_axis = range(0, stop=TOTAL_LENGTH, length=NUM_CELLS)
    v_lm_final  = sol[end][1:NUM_CELLS]
    v_icc_final = sol[end][2 * NUM_CELLS + 1:3 * NUM_CELLS]
    correlation = calculate_spatial_correlation(v_lm_final, v_icc_final)
    
    p = plot(layout=(3, 1), size=(800, 900))
    plot!(p[1], x_axis, v_lm_final, ylabel="LM Potential", title="LM Layer", legend=false)
    plot!(p[2], x_axis, v_icc_final, ylabel="ICC Potential", title="ICC Layer", legend=false)
    plot!(p[3], x_axis, correlation, xlabel="Distance from pylorus (cm)",
          ylabel="Correlation", title="Final State Cross-correlation", legend=false, ylims=(0, 1))
    
    savefig(p, "profile_correlation.svg")
end

"""
Calculate time-averaged spatial correlation between LM and ICC layers.
"""
function calculate_time_averaged_correlation(sol, timespan)
    t_indices = findall(t -> timespan[1] ≤ t ≤ timespan[2], sol.t)
    correlation_sum = zeros(NUM_CELLS)
    
    for t in sol.t[t_indices]
        state = sol(t)
        v_lm  = state[1:NUM_CELLS]
        v_icc = state[2 * NUM_CELLS + 1:3 * NUM_CELLS]
        correlation_sum .+= calculate_spatial_correlation(v_lm, v_icc)
    end
    return correlation_sum ./ length(t_indices)
end

"""
Plot the time-averaged cross-correlation.
"""
function analyze_time_averaged_correlation(sol, timespan)
    x_axis = range(0, stop=TOTAL_LENGTH, length=NUM_CELLS)
    avg_correlation = calculate_time_averaged_correlation(sol, timespan)
    
    p = plot(x_axis, avg_correlation, xlabel="Distance from pylorus (cm)",
             ylabel="Correlation", title="Time-averaged Cross-correlation", legend=false, ylims=(0, 1))
    savefig(p, "time_averaged_correlation.svg")
end

"""
Moving average implementation
"""
function movmean(x::Vector{Float64}, window::Int)
    n = length(x)
    result = zeros(n)
    half_window = window ÷ 2
    
    for i in 1:n
        start_idx = max(1, i - half_window)
        end_idx = min(n, i + half_window)
        result[i] = mean(x[start_idx:end_idx])
    end
    
    return result
end

"""
Generate space–time heatmaps for the LM and ICC layers.
"""
function analyze_spacetime(sol, timespan)
    x_axis = range(0, stop=TOTAL_LENGTH, length=NUM_CELLS)
    t_indices = findall(t -> timespan[1] ≤ t ≤ timespan[2], sol.t)
    times = sol.t[t_indices]
    v_lm_history = [sol(t)[1:NUM_CELLS] for t in times]
    v_icc_history = [sol(t)[2 * NUM_CELLS + 1:3 * NUM_CELLS] for t in times]
    
    p1 = heatmap(x_axis, times, reduce(hcat, v_lm_history)', xlabel="Distance from pylorus (cm)",
                 ylabel="Time (s)", title="Space-Time Plot - LM Layer", colorbar=:right)
    savefig(p1, "spacetime_lm.svg")
    
    p2 = heatmap(x_axis, times, reduce(hcat, v_icc_history)', xlabel="Distance from pylorus (cm)",
                 ylabel="Time (s)", title="Space-Time Plot - ICC Layer", colorbar=:right)
    savefig(p2, "spacetime_icc.svg")
end

"""
Calculate the dominant frequency (in cycles/min) from a time series.
"""
function calculate_frequency(time_series, dt)
    n_samples = length(time_series)
    freq = fftfreq(n_samples, 1/dt)
    pos_idx = freq .> 0
    freq = freq[pos_idx]
    power = abs2.(fft(time_series))[pos_idx]
    freq_cpm = freq .* 60
    return freq_cpm[argmax(power)]
end

"""
Analyze and plot the frequency content for each cell.
"""
function analyze_frequencies(sol, timespan)
    dt = sol.t[2] - sol.t[1]
    t_indices = findall(t -> timespan[1] ≤ t ≤ timespan[2], sol.t)
    time_window = sol.t[t_indices]
    
    freqs_lm = zeros(NUM_CELLS)
    freqs_icc = zeros(NUM_CELLS)
    
    for i in 1:NUM_CELLS
        ts_lm = [sol(t)[i] for t in time_window]
        ts_icc = [sol(t)[2 * NUM_CELLS + i] for t in time_window]
        freqs_lm[i] = calculate_frequency(ts_lm, dt)
        freqs_icc[i] = calculate_frequency(ts_icc, dt)
    end

    freqs_lm_smooth = movmean(freqs_lm, FREQ_MOVMEAN_WINDOW)
    freqs_icc_smooth = movmean(freqs_icc, FREQ_MOVMEAN_WINDOW)
    
    x = range(0, stop=1, length=NUM_CELLS)
    x_axis = range(0, stop=TOTAL_LENGTH, length=NUM_CELLS)
    intrinsic_freqs = [calculate_intrinsic_frequency(xi, TOTAL_LENGTH) for xi in x]
    avg_freqs = (freqs_lm_smooth + freqs_icc_smooth) ./ 2
    
    p1 = plot(x_axis, freqs_lm_smooth, label="LM Layer", linewidth=2,
              xlabel="Distance from pylorus (cm)", ylabel="Frequency (cycles min⁻¹)",
              title="Individual Layer Frequencies")
    plot!(p1, x_axis, freqs_icc_smooth, label="ICC Layer", linewidth=2)
    plot!(p1, x_axis, intrinsic_freqs, label="Intrinsic", linestyle=:dash, color=:black)
    ylims!(p1, 0, 30)
    savefig(p1, "frequencies_individual.svg")
    
    p2 = plot(x_axis, avg_freqs, label="Average", linewidth=2,
              xlabel="Distance from pylorus (cm)", ylabel="Frequency (cycles min⁻¹)",
              title="Average Layer Frequency vs Intrinsic")
    plot!(p2, x_axis, intrinsic_freqs, label="Intrinsic", linestyle=:dash, color=:black)
    ylims!(p2, 0, 30)
    savefig(p2, "frequencies_average.svg")
end

"""
Main execution
"""
sol = simulate_intestine((0, TOTAL_TIME))
println("Simulation complete. Generating analysis plots...")
analyze_final_state(sol)
analyze_time_averaged_correlation(sol, CORRELATION_TIMESPAN)
analyze_spacetime(sol, SPACETIME_TIMESPAN)
analyze_frequencies(sol, FREQUENCY_TIMESPAN)
