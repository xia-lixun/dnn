# bilinear transformation of transfer function from s-domain to z-domain
# via s = 2/T (z-1)/(z+1)
# let ζ = z^(-1) we have s = -2/T (ζ-1)/(ζ+1)
# 
#          b_m s^m + b_(m-1) s^(m-1) + ... + b_1 s + b_0
# H(s) = -------------------------------------------------
#          a_n s^n + a_(n-1) s^(n-1) + ... + a_1 s + a_0
#
# So 
#
#          b_m (-2/T)^m (ζ-1)^m / (ζ+1)^m  + ... + b_1 (-2/T) (ζ-1)/(ζ+1) + b_0 
# H(ζ) = -------------------------------------------------------------------------
#          a_n (-2/T)^n (ζ-1)^n / (ζ+1)^n  + ... + a_1 (-2/T) (ζ-1)/(ζ+1) + a_0
#
# Since we assume H(s) is rational, so n ≥ m, multiply num/den with (ζ+1)^n ans we have
#
#          b_m (-2/T)^m (ζ-1)^m (ζ+1)^(n-m)  + b_(m-1) (-2/T)^(m-1) (ζ-1)^(m-1) (ζ+1)^(n-m+1) + ... + b_1 (-2/T) (ζ-1)(ζ+1)^(n-1) + b_0 (ζ+1)^n
# H(ζ) = ---------------------------------------------------------------------------------------------------------------------------------------
#          a_n (-2/T)^n (ζ-1)^n  + a_(n-1) (-2/T)^(n-1) (ζ-1)^(n-1) (ζ+1) ... + a_1 (-2/T) (ζ-1)(ζ+1)^(n-1) + a_0 (ζ+1)^n
#
#
#         B[0] + B[1]ζ + B[2]ζ^2 + ... B[m]ζ^m
# H(ζ) = ---------------------------------------
#         A[0] + A[1]ζ + A[2]ζ^2 + ... A[n]ζ^n
using Polynomials



function bilinear(b, a, fs)

    m = size(b,1)-1
    n = size(a,1)-1
    p = Polynomials.Poly{BigFloat}(BigFloat(0))
    q = Polynomials.Poly{BigFloat}(BigFloat(0))

    br = convert(Array{BigFloat,1}, flipdim(b,1))
    ar = convert(Array{BigFloat,1}, flipdim(a,1))

    for i = m:-1:0
        p = p + (br[i+1] * (BigFloat(-2*fs)^i) * poly(convert(Array{BigFloat,1},ones(i))) * poly(convert(Array{BigFloat,1},-ones(n-i))))
    end
    for i = n:-1:0
        q = q + (ar[i+1] * (BigFloat(-2*fs)^i) * poly(convert(Array{BigFloat,1},ones(i))) * poly(convert(Array{BigFloat,1},-ones(n-i))))        
    end
    
    num = zeros(Float64,n+1)
    den = zeros(Float64,n+1)
    for i = 0:n
        num[i+1] = Float64(p[i])        
    end
    for i = 0:n
        den[i+1] = Float64(q[i])        
    end
    g = den[1]
    (num/g, den/g)
end



function convolve(a::Array{T,1}, b::Array{T,1}) where T <: Real
    m = size(a,1)
    n = size(b,1)
    l = m+n-1
    y = Array{T,1}(l)

    for i = 0:l-1
        i1 = i
        tmp = zero(T)
        for j = 0:n-1
            ((i1>=0) & (i1<m)) && (tmp += a[i1+1]*b[j+1])
            i1 -= 1
        end
        y[i+1] = tmp
    end
    y
end



# example: create a-weighting filter in z-domain
function weighting_a(fs)
    
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    p = [ ((2π*f4)^2) * (10^(A1000/20)), 0, 0, 0, 0 ]
    q = convolve(convert(Array{BigFloat,1}, [1, 4π*f4, (2π*f4)^2]), convert(Array{BigFloat,1}, [1, 4π*f1, (2π*f1)^2]))
    q = convolve(convolve(q, convert(Array{BigFloat,1}, [1, 2π*f3])),convert(Array{BigFloat,1}, [1, 2π*f2]))
    
    #(p, convert(Array{Float64,1},q))
    num_z, den_z = bilinear(p, q, fs)
end