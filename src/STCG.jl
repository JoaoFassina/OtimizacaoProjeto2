export STCG

using LinearAlgebra
using CUTEst
using NLPModels, LinearOperators, Krylov, SolverTools, SolverBenchmark,JuMP, Ipopt

function STCG(nlp;x :: AbstractVector=copy(nlp.meta.x0),
    atol :: Real=√eps(eltype(x)), rtol :: Real=√eps(eltype(x)),max_iter = 1_000_000)
    t₀ = time()
    Δt = time() - t₀

    max_time = 30
    ϵ = atol + rtol * norm(gx)
    f(x) = obj(nlp,x)
    ∇f(x) = grad(nlp,x)
    H(x) = hess(nlp,x)
    Δ = 1.0
    η = 1e-3
    iter = 0
    
    p= x .* 0 
    while norm(∇f(x)) > ϵ
        
        z = zero(∇f(x))
        r = ∇f(x)
        d = -r
        n = length(∇f(x))
        fx = f(x)
        gx = ∇f(x)
        Hx = H(x)

        if norm(r) < ϵ  
            p = z
        end
    
        for j = 1:n
            
            if dot(d, Hx*d) ≤ 0 
                model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
                @variable(model, t)
                @objective(model, Min, fx + dot(gx, z + t*d) + dot((z + t*d), Hx*(z + t*d))/2)
                for i = 1:length(z)
                    @constraint(model, (z[i] + t*d[i])^2 == Δ^2)
                end
                @constraint(model, t ≥ 0)
                optimize!(model)
                τ = value.(t)
                p = z + τ*d
            end
            α = dot(r, r)/dot(d, Hx*d)
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
            rnew = r + α*Hx*d

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
        pred = f(x) - (fx + dot(gx, p) + dot(p, Hx*p)/2) 
        
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