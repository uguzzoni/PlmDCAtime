import NLopt


""" Optimization at fixed site:
	model paramters x of objective function gradient fg!
"""

function optimize!(x::Vector{Float64}, fg!;
							verbose::Bool = false,
   							algorithm::Symbol = :LD_LBFGS,
   							lb::Vector{Float64} = fill(-50.0, length(x)),
   							ub::Vector{Float64} = fill(+50.0, length(x)),
   							xatol::Real=1e-6, xrtol::Real=1e-6,
   							fatol::Real=1e-6, frtol::Real=1e-6,
   							maxeval::Integer=10_000)
    @assert length(lb) == length(ub) == length(x)
    for i = 1:length(x) @assert lb[i] ≤ x[i] ≤ ub[i] end
    @assert xatol ≥ 0 && xrtol ≥ 0
    @assert fatol ≥ 0 && frtol ≥ 0
    @assert maxeval ≥ 0
    if iszero(xatol) && iszero(xrtol) && iszero(fatol) && iszero(frtol)
        @warn "NLoptOpts: All tolerances were set to zero. " *
              "The algorithm might never converge."
    end

    opt = NLopt.Opt(algorithm, length(x))
    NLopt.min_objective!(opt, fg!)
    NLopt.xtol_abs!(opt, xatol)
    NLopt.xtol_rel!(opt, xrtol)
    NLopt.ftol_abs!(opt, fatol)
    NLopt.ftol_rel!(opt, frtol)
    NLopt.maxeval!(opt, maxeval)
    NLopt.lower_bounds!(opt, lb)
    NLopt.upper_bounds!(opt, ub)

    optf, optx, status = NLopt.optimize!(opt, x)
    verbose && @show optf, status
    return optx, optf, status
end



function optimize(x::Matrix{Float64}, w::Workspace, beta::Vector{Float64}, v::Vector{Float64}; kwargs...)

	vecf = SharedArray{Float64}(w.N)
	fails = SharedArray{Int}(w.N)

	@assert size(x,1) == 2*w.LL

	optx = @distributed hcat for site = 1:w.N

		opx, opf, status = optimize!( x[:,site], nlopt_fg!(w, site, beta, v); kwargs...)

		if status == :FAILURE
				fails[site] = 1
				error("Optimization failed")
		end

		vecf[site] = opf
		opx
	end

	return sdata(optx), sdata(vecf), sdata(fails)
end

function nlopt_fg!(w::Workspace, site::Int, beta::Vector{Float64}, v::Vector{Float64})
	return function fg!(x::Vector{Float64}, g::Vector{Float64})
		PLsiteAndGrad!(x, g, w, site, beta, v)
	end
end
