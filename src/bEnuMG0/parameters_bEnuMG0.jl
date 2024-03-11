function initialization(init::Symbol, sample::DataSample, LL::Int, q::Int, N::Int, Jstart::Matrix{Float64})

	@assert init in [:logPi, :ParStart]

	if 	init == :ParStart
		x0 = copy(Jstart)
	elseif  init == :logPi
		x0 = initialization(sample, LL, q, N, size(Jstart,1))
	end
	x0

end

# h=log(Pi)
function initialization(sample::DataSample, LL::Int, q::Int, N::Int, size::Int; pc::Float64 = 0.05)

	@assert size=LL

	Pi, _ = BioseqUtils.compute_P(sdata(sample.Z),q,N;weight=sdata(sample.W))
	Pi = (1 - pc) * Pi .+ pc/q # add pseudocount

	par=zeros(Float64,size,N)
	par[LL-q+1:LL,:] = log.(Pi)

	return par

end

function initialization(init::Symbol, sample::Vector{DataSample}, LL::Int, q::Int, N::Int, Jstart::Matrix{Float64})

	@assert init in [:logPi, :ParStart]

	if 	init == :ParStart
		x0 = copy(Jstart)
	elseif  init == :logPi
		x0 = initialization(sample, LL, q, N, size(Jstart,1))
	end
	x0

end

# h=log(Pi)
function initialization(sample::Vector{DataSample}, LL::Int, q::Int, N::Int, size::Int; pc::Float64 = 0.05)

	@assert size==2*LL

	Pi, _ = BioseqUtils.compute_P(sdata(sample[end].Z),q,N;weight=sdata(sample[end].W))
	Pi = (1 - pc) * Pi .+ pc/q # add pseudocount
	Pi0, _ = BioseqUtils.compute_P(sdata(sample[1].Z),q,N;weight=sdata(sample[1].W))
	Pi0 = (1 - pc) * Pi0 .+ pc/q

	par=zeros(Float64,size,N)
	par[LL-q+1:LL,:] = log.(Pi)
	par[2*LL-q+1:2*LL,:] = log.(Pi0)

	return par

end



"""
	Conversion parameters to Tensor
"""
function Par2Tensor(x::Matrix{Float64}, q::Int, N::Int)

	LL = (N-1)*q*q+q
	
	@assert LL == size(x,1)

	JJ = reshape(x[1:LL-q,:], q,q,N-1,N)

	Jtensor = zeros(q,q,N,N)
    for i=1:(N-1)
        for j=(i+1):N
            Jtensor[:,:,i,j] = (JJ[:,:,j-1,i].+ permutedims(JJ[:,:,i,j],[2,1]) )/2;# not obvious but tested #J_ij as estimated from from g_i + J_ij as estimated from from g_j.
			Jtensor[:,:,j,i] = permutedims(Jtensor[:,:,i,j],[2,1]); #J_ij as estimated from from g_i + J_ij as estimated from from g_j.
	    end
    end

    htensor = fill(0.0,q,N)
    for i in 1:N
        htensor[:,i] = x[LL-q+1:LL,i]
    end


     return Jtensor, htensor

end

function compute_energy_seq(J::Array{T,4},h::Array{T,2},s::Array{Int,1})  where T<:Real


    N = size(J,3)

    e=0.0
    for i in 1:N-1
        for j in i+1:N
            e-=J[s[i],s[j],i,j];
        end
    end

    for i in 1:N
        e-=h[s[i],i];
    end

    return e
end
