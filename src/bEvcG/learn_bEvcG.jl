"""
	Computing v*G+b*E parameters
	INPUT: workspace variable with T+1 rounds (including initial library) and vectors of beta and v with T+1 elements (round 0 1 2 .. T)
	OUTPUT: G and E parameters (J and h)
"""

function  learn_par(w::Workspace, beta::Vector{Float64}, v::Vector{Float64};
						q::Int=20,
						N::Int=w.N,
						init::Symbol = :ParStart,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	if (w.T) != length(beta)
		throw(DimensionMismatch("beta elements does not match rounds"))
	end

	if (w.T) != length(v)
		throw(DimensionMismatch("beta elements does not match rounds"))
	end

	#learn model
	x = initialization(init, [w.samples[1],w.samples[w.T]], w.LL, w.q, w.N, Jstart)
	x, pslike, fail = optimize(x, w, beta, v; w.opt_args...)

	JE,hE,JG,hG = Par2Tensor(x, w.q, w.N)

	return (Jmat=x, JE=JE, hE = hE , JG = JG, hG = hG, psl=pslike)
end

"""
 Computing H = v(t) * G + β(t) * E parameters for a grid of beta and v values (including offset c)

 INPUT: data with T+1 samples and for each beta and v element a range of values
 OUTPUT: pseudolikelihood values for each grid point
"""

function  learn_par(w::Workspace, bgrid::Vector{Vector{Float64}}, vgrid::Vector{Vector{Float64}}, c::Float64;
						q::Int=20,
						N::Int=w.N,
						init::Symbol = :ParStart,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	if (w.T-2) != length(bgrid)
		throw(DimensionMismatch("beta elements does not match (number rounds - 2)"))
	end

	if (w.T-2) != length(vgrid)
		throw(DimensionMismatch("beta elements does not match (number rounds - 2)"))
	end

	for i=1:w.T-2
		@assert !(0 in vgrid[i])
	end	

	vg = Array{Array{Float64,1}}(undef,length(vgrid))

	for i=1:w.T-2
		vg[i] = [1/((1/j)+c) for j in vgrid[i]]
	end

	#learn model for each beta grid values
	x = initialization(init, [w.samples[1],w.samples[w.T]], w.LL, w.q, w.N, Jstart)

	iter_b = collect.(Iterators.product(bgrid...))
	iter_v = collect.(Iterators.product(vg...))
	psl = 0.0
	b = Vector{Float64}(undef,w.T)
	v = Vector{Float64}(undef,w.T)
	tab = Matrix{Float64}(undef,length(iter_b)*length(iter_v),2*w.T-3)

	counter=0
	for i in iter_b
		for j in iter_v
			counter += 1
			b = vcat(0.,1.,i)
			v = vcat(1/c,1/(1+c),j)
			x, pslike, fail = optimize(x, w, b, v; w.opt_args...)
			if 1 in fail
				psl = NaN
			else
				psl = sum(pslike)
			end
			tab[counter,:] = vcat(i,j,psl)
		end

	end

	return tab
end

"""
	Grid over c for model H=β*E+v*G, v=1/(t+c)
"""

function  learn_par(w::Workspace, beta::Vector{Float64}, c_scan::Vector{Float64}, idx_round::Vector{Int};
						q::Int=20,
						N::Int=w.N,
						init::Symbol = :ParStart,
						Jstart::Matrix{Float64}=zeros(Float64,2*((N-1)*q*q+q),N))

	if w.T != length(beta) != length(idx_round)
		throw(DimensionMismatch("beta elements does not match number rounds"))
	end

	v = zeros(Float64,w.T)
	psl = 0.0
	tab = Matrix{Float64}(undef,length(c_scan),2)

	counter = 0
	x = initialization(init, [w.samples[1],w.samples[w.T]], w.LL, w.q, w.N, Jstart)

	#learn model for each c value
	
	for c in c_scan
		counter += 1
		v = [1/((t-idx_round[1])/(idx_round[2]-idx_round[1])+c) for t in idx_round]
		#v = [1/(t/idx_round[1]+c) for t in idx_round]
		x, pslike, fail = optimize(x, w, beta, v; w.opt_args...)
		if 1 in fail
			psl = NaN
		else
			psl = sum(pslike)
		end
		tab[counter,:] = vcat(c,psl)
	end

	return tab
end
