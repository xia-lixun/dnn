module FORWARD
# forward propagate through the neural net
# decomposition/reconstrauction of the data under validation and test



import MAT
import HDF5
import JSON
import WAV

include("ui.jl")
include("feature.jl")
include("stft2.jl")



# tensorflow nn parameters
struct TF{T <: AbstractFloat}

    w::Array{Array{T,2},1}  # weights, w[1] is not used
    b::Array{Array{T,1},1}  # biases, b[1] is not used
    L::Int64  # layers
    width_i::Int64
    width_o::Int64

    function TF{T}(model::String) where T <: AbstractFloat
        net = MAT.matread(model)
        w = Array{Array{T,2},1}()
        b = Array{Array{T,1},1}()

        L = 4
        for i = 1:L
            push!(w, transpose(net["W$i"]))
            push!(b, vec(net["b$i"]))
        end

        wdi = size(w[1], 2)
        wdo = size(w[end], 1)

        new(w, b, L, wdi, wdo)
    end 
end




# Propagate the input data matrix through neural net
# 1. x is column major, i.e. each column is an input vector to the net 
function feedforward(model, x::AbstractArray{T,2}) where T <: AbstractFloat
    
    net = TF{Float32}(model)
    n = size(x,2)
    y = zeros(T, net.width_o, n)

    p = UI.Progress(10)
    for i = 1:n
        a = FEATURE.sigmoid.(net.w[1] * view(x,:,i) .+ net.b[1])
        for j = 2 : net.L-1
            a .= FEATURE.sigmoid.(net.w[j] * a .+ net.b[j])
        end
        y[:,i] .= net.w[net.L] * a .+ net.b[net.L]
        UI.update(p, i, n)
    end
    info("nn feed forward done.")
    y
end





# stft2() -> bm_processing() -> bm
function bm_processing(model::String, 
                       𝕏::AbstractArray{Complex{T},2}, 
                       r::Int64, 
                       t::Int64,
                       μ::AbstractArray{T,1}, σ::AbstractArray{T,1}) where T <: AbstractFloat
    x = abs.(𝕏)
    x .= (x .- μ) ./ σ
    y = FEATURE.sliding(x, r, t)
    net = TF{Float32}(model)
    bm = feedforward(net, y)
    bm
end

# do VOLA processing of a wav file
function vola_processing(specification::String, wav::String; model::String = "")
    
        s = JSON.parsefile(specification)

        root = s["root"]
        fs = s["sample_rate"]
        nfft = s["feature"]["frame_length"]
        nhp = s["feature"]["hop_length"]
        nat = s["feature"]["nat_frames"]
        ntxt = s["feature"]["frame_context"]
        assert(isodd(ntxt))
        r = div(ntxt-1,2)
        
        # get global mu and std
        stat = joinpath(root, "training", "stat.h5")
        μ = Float32.(HDF5.h5read(stat, "mu"))
        σ = Float32.(HDF5.h5read(stat, "std"))
    
        # get input data
        x, fs1 = WAV.wavread(wav)
        assert(typeof(fs)(fs1) == fs)
        x = Float32.(x)        
        𝕏, h = STFT2.stft2(view(x,:,1), nfft, nhp, STFT2.sqrthann)
    
        # reconstruct
        if isempty(model)
            y = STFT2.stft2(𝕏, h, nfft, nhp, STFT2.sqrthann)*2
        else
            𝕏 .*= bm_processing(model, 𝕏, r, nat, μ, σ)
            y = STFT2.stft2(𝕏, h, nfft, nhp, STFT2.sqrthann)*2
        end
        WAV.wavwrite(y, wav[1:end-4]*"-processed.wav", Fs=fs)
        nothing
    end










# do magnitude processing through the net
# 1. input is un-normalized col-major magnitude spectrum
# 2. output is un-normalized col-major noise-reduced magnitude spectrum
function psd_processing!(model::String, 
                         x::AbstractArray{T,2}, 
                         r::Int64,
                         t::Int64,
                         μ::AbstractArray{T,1}, σ::AbstractArray{T,1}) where T <: AbstractFloat
    x .= log.(x .+ eps())
    x .= (x .- μ) ./ σ
    y = FEATURE.sliding(x, r, t)
    nn = TF{Float32}(model)
    x .= feedforward(nn, y) .* σ .+ μ
    x .= exp.(x)
end


# do COLA processing of a wav file
function cola_processing(specification::String, wav::String; model::String = "")

    s = JSON.parsefile(specification)
    s_frame = s["feature"]["frame_size"]
    s_hop = s["feature"]["step_size"]
    s_r = s["feature"]["frame_neighbour"]
    s_t = s["feature"]["nat_size"]
    s_fs = s["sample_rate"]
    s_win = Dict("Hamming"=>FEATURE.hamming, "Hann"=>FEATURE.hann)

    # get global mu and std
    stat = joinpath(s["mix_root"], "global.h5")
    μ = Float32.(h5read(stat, "mu"))
    σ = Float32.(h5read(stat, "std"))

    # get input data
    x, fs = WAV.wavread(wav)
    assert(fs == typeof(fs)(s_fs))
    x = Float32.(x)
    
    # convert to frequency domain
    param = FEATURE.Frame1D{Int64}(s_fs, s_frame, s_hop, 0)
    nfft = s_frame
    ρ = Float32(1 / nfft)
    _cola = s_hop / sum(FEATURE.hamming(Float32, nfft))
    
    𝕏, lu = FEATURE.spectrogram(view(x,:,1), param, nfft, window=s_win[s["feature"]["window"]])
    m = size(𝕏, 2)
    y = zeros(Float32, lu)

    # reconstruct
    ImagAssert = 0.0f0
    if isempty(model)
        𝕏 = _cola * real(ifft(𝕏, 1))
    else
        # keep phase info
        𝚽 = angle.(𝕏)

        # calculate power spectra
        nfft2 = div(nfft,2)+1
        ℙ = ρ.*(abs.(view(𝕏,1:nfft2,:))).^2
        psd_processing!(model, ℙ, s_r, s_t, μ, σ)
        ℙ .= sqrt.(ℙ)./sqrt(ρ)
        ℙ = vcat(ℙ, ℙ[end-1:-1:2,:])
        𝕏 = ifft(ℙ .* exp.(𝚽 .* im), 1)
        ImagAssert = sum(imag(𝕏))
        𝕏 = _cola * real(𝕏)
    end

    for k = 0:m-1
        y[k*s_hop+1 : k*s_hop+nfft] .+= 𝕏[:,k+1]
    end
    WAV.wavwrite(y, wav[1:end-4]*"-processed.wav", Fs=s_fs)
    ImagAssert
end



# module
end