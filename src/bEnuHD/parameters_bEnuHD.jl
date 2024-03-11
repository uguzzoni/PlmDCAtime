function initialization(init::Symbol, sample::DataSample, LL::Int, q::Int, N::Int, ParStart::Matrix{Float64})

	@assert init in [:logPi, :ParStart]

	if 	init == :ParStart
		x0 = copy(ParStart)
	elseif  init == :logPi
		x0 = initialization(sample, LL, q, N, ParStart)
	end
	x0

end

# h=log(Pi)
function initialization(sample::DataSample, LL::Int, q::Int, N::Int, ParStart::Matrix{Float64}; pc::Float64 = 0.05)

	Pi, _ = BioseqUtils.compute_P(sdata(sample[end].Z),q,N;weight=sdata(sample[end].W))
	Pi = (1 - pc) * Pi .+ pc/q # add pseudocount

	ParStart[LL-q+1:LL,:] .= log.(Pi)

	return ParStart

end



"""
	Conversion parameters to Tensor
"""
function Par2Tensor(x::Matrix{Float64}, q::Int, N::Int)

    LL = (N-1)*q*q+q

	E_JJ = reshape(x[1:LL-q,:], q,q,N-1,N)

	E_Jtensor = zeros(q,q,N,N)
    for i=1:(N-1)
        for j=(i+1):N
            E_Jtensor[:,:,i,j] = (E_JJ[:,:,j-1,i].+ permutedims(E_JJ[:,:,i,j],[2,1]) )/2;# not obvious but tested #J_ij as estimated from from g_i + J_ij as estimated from from g_j.
			E_Jtensor[:,:,j,i] = permutedims(E_Jtensor[:,:,i,j],[2,1]); #J_ij as estimated from from g_i + J_ij as estimated from from g_j.
	    end
    end

    E_htensor = fill(0.0,q,N)
    for i in 1:N
        E_htensor[:,i] = x[LL-q+1:LL,i]
    end


     return E_Jtensor, E_htensor

end

function EMat2TensNoGauge(x::Matrix{Float64}, q::Int, N::Int)

    LL = (N-1)*q*q+q

    E_JJ = reshape(x[1:LL-q,:], q,q,N-1,N)


    E_Jtemp1=zeros(q,q,Int(N*(N-1)/2))
    E_Jtemp2=zeros(q,q,Int(N*(N-1)/2))
    l = 1

    for i=1:(N-1)
        for j=(i+1):N
            E_Jtemp1[:,:,l]=E_JJ[:,:,j-1,i]; #E_J_ij as estimated from from g_i.
            E_Jtemp2[:,:,l]=E_JJ[:,:,i,j]; #E_J_ij as estimated from from g_j.
            l=l+1;
        end
    end

    E_Jtensor1 = zeros(q,q,N,N)
    E_Jtensor2 = zeros(q,q,N,N)
    l = 1
    for i = 1:N-1
        for j=i+1:N
            E_Jtensor1[:,:,i,j] = E_Jtemp1[:,:,l]
            E_Jtensor2[:,:,j,i] = E_Jtemp2[:,:,l]
            l += 1
        end
    end

    E_htensor = fill(0.0,q,N)

    for i in 1:N
        E_htensor[:,i] = x[LL-q+1:LL,i]
    end

    for i in 1:N-1
        for j in i+1:N
            E_Jtensor1[:,:,j,i] = E_Jtensor1[:,:,i,j]'
            E_Jtensor2[:,:,i,j] = E_Jtensor2[:,:,j,i]'
        end
    end

    E_Jtensor = 0.5*(E_Jtensor1 + E_Jtensor2)

     return E_Jtensor, E_htensor

end
