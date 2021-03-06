export gradienteCon

using ForwardDiff
using JSOSolverTemplate
using LinearAlgebra
using CUTEst
using NLPModels, LinearOperators, Krylov, SolverTools, SolverBenchmark


function gradienteCon(nlp ;x :: AbstractVector=copy(nlp.meta.x0),
    atol :: Real=√eps(eltype(x)), rtol :: Real=√eps(eltype(x)), max_time = 30.0, max_iter = 1_000_000, η₁ = 1e-2)
    t₀ = time()
    Δt = time() - t₀
    iter = 0
    
    f(x) = obj(nlp,x)
    ∇f(x) = grad(nlp,x)
    fx = f(x)
    gx = ∇f(x)
    
    x⁺ = similar(x)
    
    ϵ = atol + rtol * norm(gx)
    
    status =:unknown
    
    resolvido = norm(gx) < ϵ
    cansado = Δt > max_time || iter > max_iter
    while !(cansado || resolvido)
        d = -gx
        slope = dot(gx,d)
        
        α = 1.0
        x⁺ = x + α * d
        f⁺ = f(x⁺)
        while f⁺ ≥ fx + η₁ * α * slope
            α = α / 2
            x⁺ = x + α * d
            f⁺ = f(x⁺)
            if α < 1e-8
                status = :small_step
                break
            end
        end
        if status != :unknown
            break
        end
        x .= x⁺
        fx = f⁺
        gx = ∇f(x)
        
        resolvido = norm(gx) < ϵ
        Δt = time() - t₀
        iter += 1
        cansado = Δt > max_time || iter > max_iter
    end
    
    if resolvido
        status = :first_order
    elseif cansado
        if Δt >max_time
            status = :max_time
        elseif iter > max_iter
            status = :max_iter
        end
    end
    
    return GenericExecutionStats(status,nlp,objective=fx,solution=x,dual_feas=norm(gx),iter = iter, elapsed_time = Δt)
end
