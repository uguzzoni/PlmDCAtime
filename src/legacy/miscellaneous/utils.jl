
function optimfunwrapper(x::Vector{Float64},g::Vector{Float64},site::Int,var::Vector{PlmVar},beta::Vector{Float64})
    g === nothing && (g = zeros(Float64, length(x)))
    return PLsiteAndGrad!(x, g, site, var, beta)
end

function optimfunwrapper(x::Vector{Float64},g::Vector{Float64},var::PlmVar,Jmat::Matrix{Float64})
    g === nothing && (g = zeros(Float64, length(x)))
    return PLbetaGrad!(x, g, var, Jmat)
end

function optimfunwrapper(x::Vector{Float64}, g::Vector{Float64},site::Int,var::PlmVar,beta::Float64)
    g === nothing && (g = zeros(Float64, length(x)))
    return PLsiteAndGrad!(x, g, site, var, beta)
end

function optimfunwrapper(x::Vector{Float64}, g::Vector{Float64},site::Int,var::PlmVar)
    g === nothing && (g = zeros(Float64, length(x)))
    return PLsiteAndGrad!(x, g, site, var)
end

function optimfunwrapper(x::Vector{Float64}, g::Vector{Float64},site::Int,var::Vector{PlmVar},beta::Vector{Float64},l::Float64)
    g === nothing && (g = zeros(Float64, length(x)))
    return PLsiteAndGrad!(x, g, site, var, beta, l)
end

function optimfunwrapper(x::Vector, g::Vector, var::Vector{PlmVar}, beta::Vector{Float64})
    g === nothing && (g = zeros(Float64, length(x)))
    return PLsiteAndGradSym!(x, g, var, beta)
end

function optimfunwrapper(x::Vector{Float64},g::Vector{Float64},site::Int,var::Vector{PlmVar},b::Vector{Float64}, v::Vector{Float64})
   g === nothing && (g = zeros(Float64, length(x)))
    return PLsiteAndGrad!(x, g, site, var, b, v)
end

#function optimfunwrapper(x::Vector{Float64},g::Vector{Float64},site::Int,var::Vector{PlmVar},bE::Vector{Float64}, bG::Vector{Float64})
#    g === nothing && (g = zeros(Float64, length(x)))
#    return PLSiteGradbEbG!(x, g, site, var, bE, bG)
#end

#Old wrapper

#function optimfunwrapper(x::Vector{Float64},g::Vector{Float64},site::Int,var::Vector{PlmVar},beta::Vector{Float64},J0::Array{Float64,4},h0::Array{Float64,2})
#    g === nothing && (g = zeros(Float64, length(x)))
#    return PLsiteAndGrad!(x, g, site, var, beta, J0, h0)
#end

#function optimfunwrapper(x::Vector{Float64}, g::Vector{Float64},var::PlmVar,G::Array{Float64,3},E0::Array{Float64,3})
#    g === nothing && (g = zeros(Float64, length(x)))
#    return PLbetaGrad!(x, g, var, G, E0)
#end

#------------------------------------

function ComputeScore(Jmat::Array{Float64,2}, N::Int, q::Int, min_separation::Int)

    JJ=reshape(Jmat[1:end-q,:], q,q,N-1,N)
    Jtemp1=zeros( q,q,Int(N*(N-1)/2))
    Jtemp2=zeros( q,q,Int(N*(N-1)/2))
    l = 1
    for i=1:(N-1)
        for j=(i+1):N
            Jtemp1[:,:,l]=JJ[:,:,j-1,i] #J_ij as estimated from from g_i.
            Jtemp2[:,:,l]=JJ[:,:,i,j]' #J_ij as estimated from from g_j.
            l=l+1
        end
    end




    hplm = fill(0.0, q,N)
    for i in 1:N
        hplm[:,i] .= Jmat[end-q+1:end,i]
    end

    Jtensor1 = inflate_matrix(Jtemp1,N)
    Jtensor2 = inflate_matrix(Jtemp2,N)
    Jplm = (Jtensor1 + Jtensor2)/2 # for the energy I do not want to gauge

    ctr = 0
    for i in 1:N-1
        for j in i+1:N
            ctr += 1
            Jtensor1[:,:,i,j] = Jtemp1[:,:,ctr]-repeat(mean(Jtemp1[:,:,ctr],dims=1),q,1)-repeat(mean(Jtemp1[:,:,ctr],dims=2),1,q) .+ mean(Jtemp1[:,:,ctr])
            Jtensor1[:,:,j,i] = Jtensor1[:,:,i,j]'
            Jtensor2[:,:,i,j] = Jtemp2[:,:,ctr]-repeat(mean(Jtemp2[:,:,ctr],dims=1),q,1)-repeat(mean(Jtemp2[:,:,ctr],dims=2),1,q) .+ mean(Jtemp2[:,:,ctr])
            Jtensor2[:,:,j,i] = Jtensor2[:,:,i,j]'
        end
    end # zerosumgauge the different tensors

    Jtensor = (Jtensor1 + Jtensor2)/2

    FN = compute_APC(Jtensor,N,q)
    score = GaussDCA.compute_ranking(FN,min_separation)
    return score, FN, Jplm, hplm
end

function compute_APC(J::Array{Float64,4},N,q)
    FN = fill(0.0, N,N)
    for i=1:N-1
        for j=i+1:N
            FN[i,j] = norm(J[1:q,1:q,i,j],2)
            FN[j,i] =FN[i,j]
        end
    end
    FN=GaussDCA.correct_APC(FN)
    return FN
end

function FrobNorm(J::Array{Float64,4},q::Int,N::Int)

    FN = zeros(Float64,N,N)

    for i = 1:N-1
        for j = i+1:N
            FN[i,j] = norm(J[:,:,i,j],2)
            FN[j,i] = FN[i,j]
        end
    end

    return FN

end



function ReadFasta(filename::AbstractString,max_gap_fraction::Real, theta::Any, remove_dups::Bool)

   Z = GaussDCA.read_fasta_alignment(filename, max_gap_fraction)
   if remove_dups
       Z, _ = GaussDCA.remove_duplicate_seqs(Z)
   end


   N, M = size(Z)
   q = round(Int,maximum(Z))

   q > 32 && error("parameter q=$q is too big (max 31 is allowed)")
   W , Meff = GaussDCA.compute_weights(Z,q,theta)
   rmul!(W, 1.0/Meff)
   Zint=round.(Int,Z)
   return W, Zint,N,M,q
end

function sumexp(vec::Array{Float64,1})
    mysum = 0.0
    @inbounds @simd for i=1:length(vec)
        mysum += exp(vec[i])
    end
    return mysum
end

function wisereldist!(Delta::Vector{Float64}, b1::Vector{Float64}, b2::Vector{Float64})

    if length(b1) != length(b2)
        throw(DimensionMismatch("Temperatures vectors must have the same length"))
    end

    for i in 1:length(b1)
        Delta[i] = abs(b1[i]-b2[i])/b2[i]
    end

end

function set_vecvar(data::NamedTuple;
                        lambdaJE::Real=0.01,
                        lambdaHE::Real=0.01,
                        lambdaJG::Real=0.01,
                        lambdaHG::Real=0.01,
                        lambdaJM::Real=0.0,
                        lambdaHM::Real=0.0,
                        #gaugecol::Int=-1,
                        weight::Symbol=:counts)

    N = data.L
    q = data.A
    T = data.T
    q2 = q*q

    vecvar=PlmVar[]

    for t = 1:T

        idxno0 = findall(data.counts[:,1,t].>0.)

        M = length(idxno0)
        Z = Array{Int8}(undef, N, M)
        Wt = Array{Float64}(undef,M)
        if weight==:counts
            Wt=data.counts[idxno0,1,t]
        elseif weight==:logcounts
                Wt=log.(data.counts[idxno0,1,t])
        else
            Wt=ones(M)
        end
        Wt=Wt./sum(Wt)

        Z = Array{Int64}(undef, N, M);
        IdxZ = Array{Int64}(undef, N, M)

        for i in 1:M
            for j in 1:N
                Z[j,i]=data.variants[idxno0[i]].sequence[j]
                IdxZ[j,i] = (j-1) * q2 + q * (Z[j,i] - 1)
            end
        end

        push!(vecvar,PlmVar(N,M,q,q2,lambdaJE,lambdaHE,lambdaJG,lambdaHG,lambdaJM,lambdaHM,Z,IdxZ,Wt))

    end

    return vecvar
end
#
# function set_vecvar1(data::NamedTuple;
#                         lambdaJE::Real=0.01,
#                         lambdaHE::Real=0.01,
#                         lambdaJG::Real=0.01,
#                         lambdaHG::Real=0.01,
#                         lambdaJM::Real=0.0,
#                         lambdaHM::Real=0.0,
#                         gaugecol::Int=-1,
#                         weight::Symbol=:counts)
#
#     N = data.L
#     q = data.A
#     T = data.T
#
#     vecvar=PlmVar[]
#
#
#     for t = 1:T
#
#         idxno0 = findall(data.counts[:,1,t].>1.)
#
#         M = length(idxno0)
#         Z = Array{Int8}(undef, N, M)
#         Wt = Array{Float64}(undef,M)
#         if weight==:counts
#             Wt=data.counts[idxno0,1,t]
#         elseif weight==:logcounts
#                 Wt=log.(data.counts[idxno0,1,t])
#         else
#             Wt=ones(M)
#         end
#         Wt=Wt./sum(Wt)
#
#         Z=zeros(Int64,N,M);
#
#         for i in 1:M
#             for j in 1:N
#                 Z[j,i]=data.variants[idxno0[i]].sequence[j]
#             end
#         end
#
#         push!(vecvar,PlmVar(N,M,q,q*q,gaugecol,lambdaJE,lambdaHE,lambdaJG,lambdaHG,lambdaJM,lambdaHM,Z,Wt))
#     end
#
#     return vecvar
# end


function FieldsMatrix2Tensor(Jmat::Matrix{Float64},N::Int,q::Int)

    JJ=reshape(Jmat[1:end-q,:], q,q,N-1,N)

    Jtemp1=zeros( q,q,Int(N*(N-1)/2))
    Jtemp2=zeros( q,q,Int(N*(N-1)/2))
    l = 1

    for i=1:(N-1)
        for j=(i+1):N
            Jtemp1[:,:,l]=JJ[:,:,j-1,i]; #J_ij as estimated from from g_i.
            Jtemp2[:,:,l]=JJ[:,:,i,j]; #J_ij as estimated from from g_j.
            l=l+1;
        end
    end


    Jtensor1 = zeros(q,q,N,N)
    Jtensor2 = zeros(q,q,N,N)
    l = 1
    for i = 1:N-1
        for j=i+1:N
            Jtensor1[:,:,i,j] = Jtemp1[:,:,l]
            Jtensor2[:,:,j,i] = Jtemp2[:,:,l]
            l += 1
        end
    end

    hplm = fill(0.0, q,N)
    htensor1 = fill(0.0, q,N)
    htensor2 = fill(0.0, q,N)

    for i in 1:N
        hplm[:,i] = Jmat[end-q+1:end,i]
    end

    for i in 1:N
        htensor1[:,i] .= hplm[:,i] .- mean(hplm[:,i])
        htensor2[:,i] .= htensor1[:,i]
        for j in 1:i-1
            htensor1[:,i] += mean(Jtensor1[:,:,i,j],dims=2) + vec(mean(Jtensor1[:,:,i,j],dims=1)) .- mean(Jtensor1[:,:,i,j])
            htensor2[:,i] += mean(Jtensor2[:,:,i,j],dims=2) + vec(mean(Jtensor2[:,:,i,j],dims=1)) .- mean(Jtensor2[:,:,i,j])
        end
        for j in i+1:N
            htensor1[:,i] += mean(Jtensor1[:,:,i,j],dims=2) + vec(mean(Jtensor1[:,:,i,j],dims=1)) .- mean(Jtensor1[:,:,i,j])
            htensor2[:,i] += mean(Jtensor2[:,:,i,j],dims=2) + vec(mean(Jtensor2[:,:,i,j],dims=1)) .- mean(Jtensor2[:,:,i,j])
        end
    end

   l = 1
   for i in 1:N-1
       for j in i+1:N
           Jtensor1[:,:,i,j] = Jtemp1[:,:,l]-repeat(mean(Jtemp1[:,:,l],dims=1),q,1)-repeat(mean(Jtemp1[:,:,l],dims=2),1,q) .+ mean(Jtemp1[:,:,l])
           Jtensor1[:,:,j,i] = Jtensor1[:,:,i,j]'
           Jtensor2[:,:,i,j] = Jtemp2[:,:,l]-repeat(mean(Jtemp2[:,:,l],dims=1),q,1)-repeat(mean(Jtemp2[:,:,l],dims=2),1,q) .+ mean(Jtemp2[:,:,l])
           Jtensor2[:,:,j,i] = Jtensor2[:,:,i,j]'
           l+=1
       end
   end

   Jtensor = (Jtensor1 + Jtensor2)/2
   htensor = (htensor1 + htensor2)/2

    return Jtensor, htensor

end

function GEMat2TensNoGauge(Jmat::Matrix{Float64},N::Int,q::Int)

    LL = (N-1)*q*q+q

    E_JJ = reshape(Jmat[1:LL-q,:], q,q,N-1,N)
    G_JJ = reshape(Jmat[LL+1:2*LL-q,:], q,q,N-1,N)



    E_Jtemp1=zeros(q,q,Int(N*(N-1)/2))
    E_Jtemp2=zeros(q,q,Int(N*(N-1)/2))
    G_Jtemp1=zeros(q,q,Int(N*(N-1)/2))
    G_Jtemp2=zeros(q,q,Int(N*(N-1)/2))
    l = 1

    for i=1:(N-1)
        for j=(i+1):N
            E_Jtemp1[:,:,l]=E_JJ[:,:,j-1,i]; #E_J_ij as estimated from from g_i.
            E_Jtemp2[:,:,l]=E_JJ[:,:,i,j]; #E_J_ij as estimated from from g_j.
            G_Jtemp1[:,:,l]=G_JJ[:,:,j-1,i]; #G_J_ij as estimated from from g_i.
            G_Jtemp2[:,:,l]=G_JJ[:,:,i,j]; #G_J_ij as estimated from from g_j.
            l=l+1;
        end
    end

    #E_Jtemp = 0.5*(E_Jtemp1 + E_Jtemp2)
    #G_Jtemp = 0.5*(G_Jtemp1 + G_Jtemp2)

    #E_Jtensor = zeros(q,q,N,N)
    #G_Jtensor = zeros(q,q,N,N)
    E_Jtensor1 = zeros(q,q,N,N)
    E_Jtensor2 = zeros(q,q,N,N)
    G_Jtensor1 = zeros(q,q,N,N)
    G_Jtensor2 = zeros(q,q,N,N)
    l = 1
    for i = 1:N-1
        for j=i+1:N
            #E_Jtensor[:,:,i,j] = E_Jtemp[:,:,l]
            #E_Jtensor[:,:,j,i] = E_Jtemp[:,:,l]'
            #G_Jtensor[:,:,i,j] = G_Jtemp[:,:,l]
            #G_Jtensor[:,:,j,i] = G_Jtemp[:,:,l]'
            E_Jtensor1[:,:,i,j] = E_Jtemp1[:,:,l]
            E_Jtensor2[:,:,j,i] = E_Jtemp2[:,:,l]
            G_Jtensor1[:,:,i,j] = G_Jtemp1[:,:,l]
            G_Jtensor2[:,:,j,i] = G_Jtemp2[:,:,l]
            l += 1
        end
    end

    E_htensor = fill(0.0,q,N)
    G_htensor = fill(0.0,q,N)

    for i in 1:N
        E_htensor[:,i] = Jmat[LL-q+1:LL,i]
        G_htensor[:,i] = Jmat[2*LL-q+1:2*LL,i]
    end

    for i in 1:N-1
        for j in i+1:N
            E_Jtensor1[:,:,j,i] = E_Jtensor1[:,:,i,j]'
            G_Jtensor1[:,:,j,i] = G_Jtensor1[:,:,i,j]'
            E_Jtensor2[:,:,i,j] = E_Jtensor2[:,:,j,i]'
            G_Jtensor2[:,:,i,j] = G_Jtensor2[:,:,j,i]'
        end
    end

    E_Jtensor = 0.5*(E_Jtensor1 + E_Jtensor2)
    G_Jtensor = 0.5*(G_Jtensor1 + G_Jtensor2)

     return E_Jtensor, E_htensor, G_Jtensor, G_htensor

end

function GEMat2Tensor(Jmat::Matrix{Float64},N::Int,q::Int)

    LL = (N-1)*q*q+q

    E_JJ = reshape(Jmat[1:LL-q,:], q,q,N-1,N)
    G_JJ = reshape(Jmat[LL+1:2*LL-q,:], q,q,N-1,N)



    E_Jtemp1=zeros(q,q,Int(N*(N-1)/2))
    E_Jtemp2=zeros(q,q,Int(N*(N-1)/2))
    G_Jtemp1=zeros(q,q,Int(N*(N-1)/2))
    G_Jtemp2=zeros(q,q,Int(N*(N-1)/2))
    l = 1

    for i=1:(N-1)
        for j=(i+1):N
            E_Jtemp1[:,:,l]=E_JJ[:,:,j-1,i]; #E_J_ij as estimated from from g_i.
            E_Jtemp2[:,:,l]=E_JJ[:,:,i,j]; #E_J_ij as estimated from from g_j.
            G_Jtemp1[:,:,l]=G_JJ[:,:,j-1,i]; #G_J_ij as estimated from from g_i.
            G_Jtemp2[:,:,l]=G_JJ[:,:,i,j]; #G_J_ij as estimated from from g_j.
            l=l+1;
        end
    end


    E_Jtensor1 = zeros(q,q,N,N)
    E_Jtensor2 = zeros(q,q,N,N)
    G_Jtensor1 = zeros(q,q,N,N)
    G_Jtensor2 = zeros(q,q,N,N)
    l = 1
    for i = 1:N-1
        for j=i+1:N
            E_Jtensor1[:,:,i,j] = E_Jtemp1[:,:,l]
            E_Jtensor2[:,:,j,i] = E_Jtemp2[:,:,l]
            G_Jtensor1[:,:,i,j] = G_Jtemp1[:,:,l]
            G_Jtensor2[:,:,j,i] = G_Jtemp2[:,:,l]
            l += 1
        end
    end

    E_hplm = fill(0.0,q,N)
    G_hplm = fill(0.0,q,N)

    for i in 1:N
        E_hplm[:,i] = Jmat[LL-q+1:LL,i]
        G_hplm[:,i] = Jmat[2*LL-q+1:2*LL,i]
    end

    E_htensor1 = fill(0.0,q,N)
    E_htensor2 = fill(0.0,q,N)
    G_htensor1 = fill(0.0,q,N)
    G_htensor2 = fill(0.0,q,N)

    for i in 1:N
        E_htensor1[:,i] .= E_hplm[:,i] .- mean(E_hplm[:,i])
        E_htensor2[:,i] .= E_htensor1[:,i]
        G_htensor1[:,i] .= G_hplm[:,i] .- mean(G_hplm[:,i])
        G_htensor2[:,i] .= G_htensor1[:,i]
        for j in 1:i-1
            E_htensor1[:,i] += mean(E_Jtensor1[:,:,i,j],dims=2) + vec(mean(E_Jtensor1[:,:,i,j],dims=1)) .- mean(E_Jtensor1[:,:,i,j])
            E_htensor2[:,i] += mean(E_Jtensor2[:,:,i,j],dims=2) + vec(mean(E_Jtensor2[:,:,i,j],dims=1)) .- mean(E_Jtensor2[:,:,i,j])
            G_htensor1[:,i] += mean(G_Jtensor1[:,:,i,j],dims=2) + vec(mean(G_Jtensor1[:,:,i,j],dims=1)) .- mean(G_Jtensor1[:,:,i,j])
            G_htensor2[:,i] += mean(G_Jtensor2[:,:,i,j],dims=2) + vec(mean(G_Jtensor2[:,:,i,j],dims=1)) .- mean(G_Jtensor2[:,:,i,j])
        end
        for j in i+1:N
            E_htensor1[:,i] += mean(E_Jtensor1[:,:,i,j],dims=2) + vec(mean(E_Jtensor1[:,:,i,j],dims=1)) .- mean(E_Jtensor1[:,:,i,j])
            E_htensor2[:,i] += mean(E_Jtensor2[:,:,i,j],dims=2) + vec(mean(E_Jtensor2[:,:,i,j],dims=1)) .- mean(E_Jtensor2[:,:,i,j])
            G_htensor1[:,i] += mean(G_Jtensor1[:,:,i,j],dims=2) + vec(mean(G_Jtensor1[:,:,i,j],dims=1)) .- mean(G_Jtensor1[:,:,i,j])
            G_htensor2[:,i] += mean(G_Jtensor2[:,:,i,j],dims=2) + vec(mean(G_Jtensor2[:,:,i,j],dims=1)) .- mean(G_Jtensor2[:,:,i,j])
        end
    end

    l = 1
    for i in 1:N-1
        for j in i+1:N
            E_Jtensor1[:,:,i,j] = E_Jtemp1[:,:,l]-repeat(mean(E_Jtemp1[:,:,l],dims=1),q,1)-repeat(mean(E_Jtemp1[:,:,l],dims=2),1,q) .+ mean(E_Jtemp1[:,:,l])
            E_Jtensor1[:,:,j,i] = E_Jtensor1[:,:,i,j]'
            G_Jtensor1[:,:,i,j] = G_Jtemp1[:,:,l]-repeat(mean(G_Jtemp1[:,:,l],dims=1),q,1)-repeat(mean(G_Jtemp1[:,:,l],dims=2),1,q) .+ mean(G_Jtemp1[:,:,l])
            G_Jtensor1[:,:,j,i] = G_Jtensor1[:,:,i,j]'
            E_Jtensor2[:,:,i,j] = E_Jtemp2[:,:,l]-repeat(mean(E_Jtemp2[:,:,l],dims=1),q,1)-repeat(mean(E_Jtemp2[:,:,l],dims=2),1,q) .+ mean(E_Jtemp2[:,:,l])
            E_Jtensor2[:,:,j,i] = E_Jtensor2[:,:,i,j]'
            G_Jtensor2[:,:,i,j] = G_Jtemp2[:,:,l]-repeat(mean(G_Jtemp2[:,:,l],dims=1),q,1)-repeat(mean(G_Jtemp2[:,:,l],dims=2),1,q) .+ mean(G_Jtemp2[:,:,l])
            G_Jtensor2[:,:,j,i] = G_Jtensor2[:,:,i,j]'
            l+=1
        end
    end

    E_Jtensor = (E_Jtensor1 + E_Jtensor2)/2
    E_htensor = (E_htensor1 + E_htensor2)/2
    G_Jtensor = (G_Jtensor1 + G_Jtensor2)/2
    G_htensor = (G_htensor1 + G_htensor2)/2

     return E_Jtensor, E_htensor, G_Jtensor, G_htensor

end

function GMEMat2TensNoGauge(Jmat::Matrix{Float64},N::Int,q::Int)

    LL = (N-1)*q*q+q

    E_JJ = reshape(Jmat[1:LL-q,:], q,q,N-1,N)
    G_JJ = reshape(Jmat[LL+1:2*LL-q,:], q,q,N-1,N)
    M_JJ = reshape(Jmat[2*LL+1:3*LL-q,:], q,q,N-1,N)



    E_Jtemp1=zeros(q,q,Int(N*(N-1)/2))
    E_Jtemp2=zeros(q,q,Int(N*(N-1)/2))
    G_Jtemp1=zeros(q,q,Int(N*(N-1)/2))
    G_Jtemp2=zeros(q,q,Int(N*(N-1)/2))
    M_Jtemp1=zeros(q,q,Int(N*(N-1)/2))
    M_Jtemp2=zeros(q,q,Int(N*(N-1)/2))
    l = 1

    for i=1:(N-1)
        for j=(i+1):N
            E_Jtemp1[:,:,l]=E_JJ[:,:,j-1,i]; #E_J_ij as estimated from from g_i.
            E_Jtemp2[:,:,l]=E_JJ[:,:,i,j]; #E_J_ij as estimated from from g_j.
            G_Jtemp1[:,:,l]=G_JJ[:,:,j-1,i]; #G_J_ij as estimated from from g_i.
            G_Jtemp2[:,:,l]=G_JJ[:,:,i,j]; #G_J_ij as estimated from from g_j.
            M_Jtemp1[:,:,l]=G_JJ[:,:,j-1,i]; #G_J_ij as estimated from from g_i.
            M_Jtemp2[:,:,l]=G_JJ[:,:,i,j]; #G_J_ij as estimated from from g_j.
            l=l+1;
        end
    end

    #E_Jtemp = 0.5*(E_Jtemp1 + E_Jtemp2)
    #G_Jtemp = 0.5*(G_Jtemp1 + G_Jtemp2)

    #E_Jtensor = zeros(q,q,N,N)
    #G_Jtensor = zeros(q,q,N,N)
    E_Jtensor1 = zeros(q,q,N,N)
    E_Jtensor2 = zeros(q,q,N,N)
    G_Jtensor1 = zeros(q,q,N,N)
    G_Jtensor2 = zeros(q,q,N,N)
    M_Jtensor1 = zeros(q,q,N,N)
    M_Jtensor2 = zeros(q,q,N,N)
    l = 1
    for i = 1:N-1
        for j=i+1:N
            #E_Jtensor[:,:,i,j] = E_Jtemp[:,:,l]
            #E_Jtensor[:,:,j,i] = E_Jtemp[:,:,l]'
            #G_Jtensor[:,:,i,j] = G_Jtemp[:,:,l]
            #G_Jtensor[:,:,j,i] = G_Jtemp[:,:,l]'
            E_Jtensor1[:,:,i,j] = E_Jtemp1[:,:,l]
            E_Jtensor2[:,:,j,i] = E_Jtemp2[:,:,l]
            G_Jtensor1[:,:,i,j] = G_Jtemp1[:,:,l]
            G_Jtensor2[:,:,j,i] = G_Jtemp2[:,:,l]
            M_Jtensor1[:,:,i,j] = G_Jtemp1[:,:,l]
            M_Jtensor2[:,:,j,i] = G_Jtemp2[:,:,l]
            l += 1
        end
    end

    E_htensor = fill(0.0,q,N)
    G_htensor = fill(0.0,q,N)
    M_htensor = fill(0.0,q,N)

    for i in 1:N
        E_htensor[:,i] = Jmat[LL-q+1:LL,i]
        G_htensor[:,i] = Jmat[2*LL-q+1:2*LL,i]
        M_htensor[:,i] = Jmat[3*LL-q+1:3*LL,i]
    end

    for i in 1:N-1
        for j in i+1:N
            E_Jtensor1[:,:,j,i] = E_Jtensor1[:,:,i,j]'
            G_Jtensor1[:,:,j,i] = G_Jtensor1[:,:,i,j]'
            M_Jtensor1[:,:,j,i] = M_Jtensor1[:,:,i,j]'
            E_Jtensor2[:,:,i,j] = E_Jtensor2[:,:,j,i]'
            G_Jtensor2[:,:,i,j] = G_Jtensor2[:,:,j,i]'
            M_Jtensor2[:,:,i,j] = M_Jtensor2[:,:,j,i]'
        end
    end

    E_Jtensor = 0.5*(E_Jtensor1 + E_Jtensor2)
    G_Jtensor = 0.5*(G_Jtensor1 + G_Jtensor2)
    M_Jtensor = 0.5*(M_Jtensor1 + M_Jtensor2)

     return E_Jtensor, E_htensor, G_Jtensor, G_htensor, M_Jtensor, M_htensor

end

function inflate_matrix(J::Array{Float64,3},N)
    q,q,NN = size(J)

    @assert (N*(N-1))>>1 == NN

    Jt = zeros(q,q,N,N)
    ctr = 0
    for i in 1:N-1
        for j in i+1:N
            ctr += 1
            Jt[:,:,i,j] = J[:,:,ctr]
            Jt[:,:,j,i] = J[:,:,ctr]'
        end
    end
    return Jt
end


function ComputePSL(Jmat::Matrix{Float64}, var::Vector{PlmVar}, beta::Vector{Float64})

    N = var[1].N
    q = var[1].q
    T = size(var,1)
    psl = zeros(Float64,T)
    vecene = zeros(Float64,q)
    reg = 0.0

    for t=1:T

        M = var[t].M
        Z = sdata(var[t].Z)
        W = sdata(var[t].W)

        for a=1:M
            for site=1:N
                fillvecene!(vecene,Jmat[:,site],site,a,var[t],beta[t])
                lnorm = log(sumexp(vecene))
                psl[t] -= W[a] * (vecene[Z[site,a]] - lnorm)
            end
        end
    end

    #for i=1:N
    #    reg += L2norm_asymEG(Jmat[:,i],var[1])
    #end

    #psl .+= reg/T

    return psl

end

#function ComputePSLtens()
#end




function logsumexp(X::AbstractArray{T}) where {T<:Real}
    isempty(X) && return log(zero(T))
    u = maximum(X)
    isfinite(u) || return float(u)
    let u=u # avoid https://github.com/JuliaLang/julia/issues/15276
        u + log(sum(x -> exp(x-u), X))
    end
end
