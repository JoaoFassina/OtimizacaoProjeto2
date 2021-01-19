export bfgs, ldlt1, bfgs_update!, solve_ldlt
using CUTEst
using NLPModels, LinearOperators, Krylov, SolverTools, SolverBenchmark,JuMP, Ipopt

function bfgs(nlp; tol = 1e-5, max_iter = 1000, max_time = 3)

  x = copy(nlp.meta.x0)
  f(x) = obj(nlp,x)
  fx = f(x)
  ef = 0
  k = 0
  t_inicial = time()
  t_total = 0.0
  n = size(x,1)
  g(x)= grad(nlp,x)
  gx = g(x)
  d = -1*gx
  B = Matrix{Float64}(I, n, n) #eye(n)
  
  status =:unknown

  while norm(gx) > tol

    t=1.0;
    p=dot(gx,d);
    #achar o passo - wolfe
    while (f(x+t*d) > fx + 1e-4*t*p || dot(g(x+t*d),d) < 0.9*p) && t >= eps(Float64)
      t=t*0.9
    end
    s = t*d
    x = x + s
    fx = f(x)
    g_prox = g(x)
    y = g_prox - gx
    bfgs_update!(B, s, y)
    L,D = ldlt1(B)
    d = solve_ldlt(L,D,-g_prox) # LDLt * d = - grad
    gx = g_prox
    k = k + 1

    if k >= max_iter
      ef = 1
      status = :max_iter
      break
    end
    t_total = time() - t_inicial
    if t_total >= max_time
      status = :max_time
      ef = 2
      break
    end

  end #do while principal
  status = :first_order
  return GenericExecutionStats(status,nlp,objective=f(x),solution=x,dual_feas=norm(g(x)),iter = k, elapsed_time = t_total)
end

#-----------------------------------------------------------------------------

function bfgs_update!(B, s, y) #s e y é o p e o q do bfgs

  (n,n)=size(B)
  a=dot(y,s)
  v=B*s
  b=dot(s,v)

  for i=1:n
    for j =1:i
      B[i,j] = B[i,j] + y[i]*y[j]/a  - v[i]*v[j]/b
    end
  end

  #aproveitar a simetria de B
  for i=1:n-1
    for j=i+1:n
      B[i,j] = B[j,i]
    end
  end

end

#-----------------------------------------------------------------------------

function ldlt1(A)

  (n,n) = size(A)
  L = Matrix{Float64}(I, n, n)
  D = zeros(n)


  for j=1:n
    soma=0.0
    for k=1:j-1
      soma = soma + D[k]*(L[j,k]^2)
    end
    D[j] = A[j,j] - soma

    for i=j+1:n
      soma=0.0
      for k=1:j-1
        soma = soma + D[k]*L[i,k]*L[j,k]
      end
      L[i,j] = (A[i,j] - soma)/D[j]
    end
  end

  return L,D
end

#-----------------------------------------------------------------------------

function solve_ldlt(L, D, b)

#Lz=b
  (n,n)=size(L)
  x=zeros(n)

  for i=1:n
    soma=0.0
    for k=1:i-1
      soma = soma + x[k]*L[i,k]
    end
    x[i] = b[i] - soma
  end

#Dy=z

  for i=1:n
    x[i] = x[i]/D[i]
  end

#L'x=y

  for i=n:-1:1
    soma=0.0
    for k=i+1:n
      soma = soma + x[k]*L[k,i]
    end
    x[i] = x[i] - soma
  end

  return x
end
