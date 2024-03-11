function emdca_sym(filename::AbstractString;
                decimation::Bool=false,
                fracmax::Float64 = 0.3,
                fracdec::Float64 = 0.1,
                blockdecimate::Bool=true, # decimate on a per-block base (all J[:,:,i,j] = 0)
                remove_dups::Bool = true,
                min_separation::Int = 1,
                max_gap_fraction::Real = 0.9, 
                theta = :auto, 
                lambdaJ::Real=0.0000008, 
                lambdaH::Real=0.0000008,
                gaugecol::Int=-1,
                epsconv::Real=1.0e-25,
                maxit::Int=100000,
                verbose::Bool=true,
                method::Symbol=:LD_LBFGS)

    plmalg = PlmAlg(method,verbose, epsconv ,maxit, nothing)
    gaugecol >= 1 && println("Warning: gaugecol not implemented. Proceeding ...")

    vecvar , T = ReadJLD(filename,lambdaJ, lambdaH)

    
    if decimation == false
        Jmat, entropy = MinimizeCESym(plmalg,vecvar,T) 
    # else
    #     if blockdecimate
    #         decvar = DecVar{1}(fracdec, fracmax, blockdecimate, ones(Bool, binomial(N,2)))
    #     else
    #         decvar = DecVar{1}(fracdec, fracmax, blockdecimate, ones(Bool, binomial(N,2)*q*q+N*q))        
    #     end
    #     Jmat, pseudolike = DecimateSym!(plmvar, plmalg, decvar)
    end
    score, Jtens, htens = ComputeScoreSym(Jmat, vecvar, min_separation)
    return output = PlmOut{3}(entropy, Jtens, htens, score)    
end
    
function MinimizeCESym(alg::PlmAlg, var::Vector{PlmVar}, T)
    N  = var[1].N
    q  = var[1].q
    q2 = var[1].q2
    
    Nc2 = binomial(N,2)
    LL  = Nc2 * q2  + N * q 

    #x0 = zeros(Float64, LL)
    #x0 = [ones(Float64, LL-N*q)*0.1;ones(Float64, N*q)*0.01]
    #x0 = -rand(collect(0:10), LL)/100
    x0 = ones(Float64, LL)*0.01

    opt = Opt(alg.method, length(x0))
    
    ftol_abs!(opt, alg.epsconv)
    maxeval!(opt, alg.maxit)
    min_objective!(opt, (x,g)->optimfunwrapper_CE(x,g,var,T))
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    @printf("pl = %.4f\t time = %.4f\t exit status = ", minf, elapstime)
    println(ret)
    
    return minx, minf
end

function ComputeScoreSym(Jvec::Array{Float64,1}, var::Vector{PlmVar}, min_separation::Int)


    LL = length(Jvec)
    N=var[1].N
    q=var[1].q
    Nc2 = binomial(N,2)

    Jtens=reshape(Jvec[1:LL-N*q],q,q,Nc2)
    htens=fill(0.0,q,N)
    J = zeros(q,q,Nc2)

    htens=reshape(Jvec[LL-N*q + 1:end],q,N)
    
    for l=1:Nc2
        J[:,:,l] = Jtens[:,:,l] - repeat(mean(Jtens[:,:,l],dims=1),q,1)-repeat(mean(Jtens[:,:,l],dims=2),1,q) .+ mean(Jtens[:,:,l])
    end
    
    
    FN = zeros(Float64, N,N)
    l = 1
    for i=1:N-1
        for j=i+1:N
            FN[i,j] = norm(J[:,:,l],2)
            FN[j,i] = FN[i,j]
            l+=1
        end
    end
    FN = GaussDCA.correct_APC(FN)  
    score = GaussDCA.compute_ranking(FN,min_separation)
    return score, Jtens,htens
end

function CEsiteAndGradSym!(vecJ::Array{Float64,1}, grad::Array{Float64,1}, plmvar::Vector{PlmVar},T)

    LL = length(vecJ)
    q2 = plmvar[1].q2
    q = plmvar[1].q
    N = plmvar[1].N
    lambdaJ = plmvar[1].lambdaJ
    lambdaH = plmvar[1].lambdaH
    
    entropy = 0.0

    for i=1:LL-N*q
        grad[i] = 2.0 * vecJ[i] * lambdaJ
    end
    for i=(LL-N*q + 1):LL
        grad[i] = 4.0 * vecJ[i] * lambdaH
    end
    
    #println("CEsiteandgrad")

    for t in T

        M = plmvar[t].M
        Z = plmvar[t].Z
        W = plmvar[t].W

        vecen = zeros(Float64,M)

        gradexp = zeros(Float64,length(grad))
        gradsimple = zeros(Float64,length(grad))

        for a = 1:M
            
            for site = 1:N
                
                vecen[a] += t*minus_energy(vecJ,Z[:,a],site,q,N)
                
            end
            ComputePatternCESym!(vecen[a], gradexp, gradsimple, vecJ, Z[:,a], N, q, q2,t)
        end
        gradexp .= gradexp./(sumexp(vecen))
        grad .+= gradexp
        grad .+= gradsimple./M
        #println(grad[1:10])
        #println(vecJ[1:10])
        entropy += log(sumexp(vecen)/M) 
        entropy -= sum(vecen)/M
    end


    #println("endCEgrad")

    entropy += L2norm_sym(vecJ, plmvar) 
    return entropy 
end

function ComputePatternCESym!(en, gradexp::Array{Float64,1}, gradsimple::Array{Float64,1}, vecJ::Array{Float64,1}, Z::Array{Int,1}, N::Int, q::Int, q2::Int, t)
    

    #vecene = zeros(Float64,q)
    #expvecenesunorm = zeros(Float64,q)

    offset = mygetindex(N-1, N, q, q, N, q, q2) 
    
    #println("computepattern")

    @inbounds for site=1:N    # site < i 
        #fillvecenesym!(vecene, vecJ, Z, site, q,N,t)        
        #norm = sumexp(vecene)
        #expvecenesunorm = exp.(vecene .- log(norm))
        #pseudolike -= Wa * ( vecene[Z[site]] - log(norm) )
    for i = 1:(site-1)
            #for s = 1:q
            #    grad[ mygetindex(i, site, Z[i], s, N, q, q2) ] += 0.5 * Wa * expvecenesunorm[s] * t
            #end
        gradexp[ mygetindex(i, site , Z[i], Z[site],  N,q,q2)] += t * exp(en) * 0.5
        gradsimple[ mygetindex(i, site , Z[i], Z[site],  N,q,q2)] -= t * 0.5
    end
    for i = (site+1):N 
            #for s = 1:q
            #    grad[ mygetindex(site, i , s,  Z[i], N,q,q2) ] += 0.5 * Wa * expvecenesunorm[s] * t
            #end
        gradexp[ mygetindex( site,i , Z[site], Z[i],  N,q,q2)] += t * exp(en) * 0.5
        gradsimple[ mygetindex(site,i , Z[site], Z[i],  N,q,q2)] -= t * 0.5
        #gradexp[ mygetindex(i, site , Z[i], Z[site],  N,q,q2)] += t * exp(en) * 0.5
        #gradsimple[ mygetindex(i, site , Z[i], Z[site],  N,q,q2)] -= t * 0.5
    end
    gradexp[ offset + Z[site] ] += t * exp(en)
    gradsimple[ offset + Z[site] ] -= t
    offset += q 
    end
end


function minus_energy(vecJ::Array{Float64,1}, Z::Array{Int64,1}, site::Int, q::Int ,N::Int)
    q2 = q*q
    offset::Int = 0
    scra::Float64 = 0.0
    
    for i=1:(site-1)
        scra += vecJ[ mygetindex(i, site, Z[i], Z[site],  N, q, q2)]
    end
    for i = (site+1):N
        scra += vecJ[ mygetindex(site, i, Z[site], Z[i], N, q, q2)]
        #scra += vecJ[ mygetindex(i, site, Z[i], Z[site],  N, q, q2)]
    end # End sum_i \neq site J

    scra *= 0.5 

    offset = mygetindex(N-1, N, q, q, N, q, q2)  + ( site - 1) * q  # last J element + (site-1)*q 
    scra += vecJ[offset + Z[site]] # sum H 


    return scra
end

function L2norm_sym(vec::Array{Float64,1}, var::Vector{PlmVar})

    q = var[1].q    
    N = var[1].N
    lambdaJ = var[1].lambdaJ
    lambdaH = var[1].lambdaH

    LL = length(vec)


    mysum1 = 0.0
    @inbounds @simd for i=1:(LL-N*q)
        mysum1 += vec[i] * vec[i]
    end
    mysum1 *= lambdaJ

    mysum2 = 0.0
    @inbounds @simd for i=(LL - N*q + 1):LL
        mysum2 += vec[i] * vec[i]
    end
    mysum2 *= 2lambdaH
    
    return mysum1+mysum2
end


function mygetindex( i::Int, j::Int, coli::Int, colj::Int, N::Int, q::Int, q2::Int)        
    offset_i = ( (i-1) * N  - ( (i * ( i -1 ) ) >> 1 ) ) * q2 # (i-1) N q2 + i (i-1) q2 / 2  
    offset_j = (j - i - 1 ) * q2
    return offset_i + offset_j + coli + q * (colj - 1)
end
