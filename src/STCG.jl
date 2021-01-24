export STCG

using LinearAlgebra
using CUTEst
using NLPModels, LinearOperators, Krylov, SolverTools, SolverBenchmark,JuMP, Ipopt

function STCG(nlp;x :: AbstractVector=copy(nlp.meta.x0),
    atol :: Real=√eps(eltype(x)), rtol :: Real=√eps(eltype(x)),max_iter = 1_000_000, max_time = 30.0)
    t₀ = time()
    Δt = time() - t₀

    f(x) = obj(nlp,x)
    ∇f(x) = grad(nlp,x)
    gx = ∇f(x)
    H(x) = hess(nlp,x) 
    
    Δ = 1.0
    η = 1e-3
    iter = 0
    ϵ = atol + rtol * norm(gx)

    status =:unknown
    
    p = x .* 0 
    z = zero(∇f(x))

    
    while norm(∇f(x)) > ϵ
        
        
        r = ∇f(x)
        d = -r
        n = length(∇f(x))
        fx = f(x)
        gx = ∇f(x)
        Hx = H(x)
        D = Hx'
        ĝx = inv(D') * gx
        B̂x = inv(D') * Hx * inv(D)

        if norm(r) < ϵ  
            p = z
        end
    
        for j = 1:n
            
            if dot(d, B̂x*d) ≤ 0 
                model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
                @variable(model, t)
                @objective(model, Min, fx + dot(ĝx, z + t*d) + dot((z + t*d), B̂x*(z + t*d))/2)
                for i = 1:length(z)
                    @constraint(model, (z[i] + t*d[i])^2 == Δ^2)
                end
                @constraint(model, t ≥ 0)
                optimize!(model)
                τ = value.(t)
                p = z + τ*d
            end
            α = dot(r, r)/dot(d, B̂x*d)
            znew = z + α*d
            
            #end 1st test
            
            if norm((znew)) ≥ Δ
                model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
                @variable(model, t)
                for i = 1:length(z)
                    @objective(model, Min, z[i] + t*d[i])
                    @constraint(model, (z[i] + t*d[i])^2 == Δ^2)
                end
                @constraint(model, t ≥ 0)
                optimize!(model)
                τ = value.(t) 
                p = z + τ*d
            end
            rnew = r + α*B̂x*d

            #end 2nd test

            if norm((rnew)) < ϵ 
                p = znew
            end
            
            βnew = dot(rnew, rnew)/dot(r, r)
            dnew = -rnew + βnew*d 

            #end 3rd test

            z, r, d = znew, rnew, dnew
        end
        
        
        
        ared = f(x) - f(x + p)
        pred = f(x) - (fx + dot(ĝx, p) + dot(p, B̂x*p)/2) 
        
        ρ = ared/pred

        if norm(ρ) < 0.25
            Δ = 0.25*Δ
        elseif norm(ρ) > 0.75 && norm(ρ) == Δ
            Δ = min(2*Δ, ϵ)
        else 
             Δ = Δ
        end
        
        if norm(ρ) > η
            x = x + p
        else
            x = x
        end

        Δt = time() - t₀
        iter += 1 
       
        if iter > max_iter
            status = :max_iter
            break
        end

        if Δt > max_time
            status = :max_time
            break
        end

    end

    if norm(∇f(x)) <= ϵ
        status = :first_order
    elseif (Δt > max_time || iter > max_iter)
        if Δt > max_time
            status = :max_time
        elseif iter > max_iter
            status = :max_iter
        end
    end
    return GenericExecutionStats(status,nlp,objective=f(x),solution=x,dual_feas=norm(∇f(x)),iter = iter, elapsed_time = Δt)
end