function initialization(init::Symbol, sample::Vector{DataSample}, LL::Int, q::Int, N::Int, Jstart::Matrix{Float64})

	@assert init in [:logPi, :ParStart]

	if 	init == :ParStart
		x0 = copy(Jstart)
	elseif  init == :logPi
		x0 = initialization(sample, LL, q, N)
	end
	x0

end

# h=log(Pi)
function initialization(sample::Vector{DataSample}, LL::Int, q::Int, N::Int; pc::Float64 = 0.05)

	Pi, _ = BioseqUtils.compute_P(sdata(sample[end].Z),q,N;weight=sdata(sample[end].W))
	Pi = (1 - pc) * Pi .+ pc/q # add pseudocount
	Pi0, _ = BioseqUtils.compute_P(sdata(sample[1].Z),q,N;weight=sdata(sample[1].W))
	Pi0 = (1 - pc) * Pi0 .+ pc/q # add pseudocount

	par=zeros(Float64,2*LL,N)
	par[LL-q+1:LL,:] = log.(Pi)
	par[2*LL-q+1:LL,:] = log.(Pi0)

	return par

end



"""
	Conversion parameters to Tensor
"""
function Par2Tensor(x::Matrix{Float64}, q::Int, N::Int)

    LL = (N-1)*q*q+q

	E_JJ = reshape(x[1:LL-q,:], q,q,N-1,N)
    G_JJ = reshape(x[LL+1:2*LL-q,:], q,q,N-1,N)

	E_Jtensor = zeros(q,q,N,N)
	G_Jtensor = zeros(q,q,N,N)
    for i=1:(N-1)
        for j=(i+1):N
            E_Jtensor[:,:,i,j] = (E_JJ[:,:,j-1,i].+ permutedims(E_JJ[:,:,i,j],[2,1]) )/2;# not obvious but tested #J_ij as estimated from from g_i + J_ij as estimated from from g_j.
			E_Jtensor[:,:,j,i] = permutedims(E_Jtensor[:,:,i,j],[2,1]); #J_ij as estimated from from g_i + J_ij as estimated from from g_j.
			G_Jtensor[:,:,i,j] = (G_JJ[:,:,j-1,i].+ permutedims(G_JJ[:,:,i,j],[2,1]) )/2
			G_Jtensor[:,:,j,i] = permutedims(G_Jtensor[:,:,i,j],[2,1]);
	    end
    end

    E_htensor = fill(0.0,q,N)
	G_htensor = fill(0.0,q,N)
    for i in 1:N
        E_htensor[:,i] = x[LL-q+1:LL,i]
		G_htensor[:,i] = x[2*LL-q+1:2*LL,i]
    end


     return E_Jtensor, E_htensor, G_Jtensor, G_htensor

end
