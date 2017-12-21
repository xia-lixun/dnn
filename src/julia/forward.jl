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



# tensorflow nn parameters initialized from the mat file generated
struct TF{T <: AbstractFloat}

    w::Array{Array{T,2},1}  # weights, w[1] is not used
    b::Array{Array{T,1},1}  # biases, b[1] is not used
    L::Int64  # layers
    width_i::Int64
    width_h::Int64
    width_o::Int64

    function TF{T}(model::String) where T <: AbstractFloat
        L = 4
        net = MAT.matread(model)
        w = Array{Array{T,2},1}(L)
        b = Array{Array{T,1},1}(L)

        for i = 1:L
            w[i] = transpose(net["W$i"])
            b[i] = vec(net["b$i"])
        end

        wdi = size(w[1], 2)
        wdh = size(w[1], 1)
        wdo = size(w[end], 1)

        new(w, b, L, wdi, wdh, wdo)
    end 
end



# Propagate the input data matrix through neural net
# 1. x is column major, i.e. each column is an input vector to the net 
function feedforward(nn::TF{T}, x::AbstractArray{T,2}) where T <: AbstractFloat
    
    # nn = TF{Float32}(model)
    # assert(nn.width_i == size(x,1))
    n = size(x,2)
    y = zeros(T, nn.width_o, n)
    a = zeros(T, nn.width_h)

    for i = 1:n
        a .= FEATURE.sigmoid.(nn.w[1] * view(x,:,i) .+ nn.b[1])
        for j = 2 : nn.L-1
            a .= FEATURE.sigmoid.(nn.w[j] * a .+ nn.b[j])
        end
        y[:,i] .= FEATURE.sigmoid.(nn.w[nn.L] * a .+ nn.b[nn.L])
    end
    y
end





# stft2() -> bm_inference() -> bm
function bm_inference(nn::TF{T}, 
                      𝕏::AbstractArray{Complex{T},2}, 
                      r::Int64, 
                      t::Int64,
                      μ::AbstractArray{T,1}, σ::AbstractArray{T,1}) where T <: AbstractFloat
    x = abs.(𝕏)
    x .= (x .- μ) ./ σ
    y = FEATURE.sliding(x, r, t)
    bm = feedforward(nn, y)
    bm
end

# do VOLA processing of a wav file
# if nn model is provided, bm_inference() will be used for bm estimate
# if no nn model is provided, it searches the training and test spectrum.h5 to see if wav
# is located in: yes -> reference bm is used for voice reconstruction;
#                no  -> reconstruct the mix back so a passing-through is achieved.
function vola_processing(specification::String, wav::String; model::String = "")
    
        s = JSON.parsefile(specification)
        fs = s["sample_rate"]
        root = s["root"]
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

        
        function bm_reference()
            # load the train/test spectrum+bm dataset
            tid = HDF5.h5open(joinpath(root,"training","spectrum.h5"),"r")
            vid = HDF5.h5open(joinpath(root,"test","spectrum.h5"),"r")
            
            tbi = contains.(names(tid),basename(wav))
            vbi = contains.(names(vid),basename(wav))
            sumt = sum(tbi)
            sumv = sum(vbi)
            if sumt + sumv == 1
                info("found in training/test wav files, use optimal bm")
                hit = sumt > sumv ? names(tid)[tbi][1] : names(vid)[vbi][1]
                bm = sumt > sumv ?  read(tid[hit]["bm"]) : read(vid[hit]["bm"])
                close(vid)
                close(tid)
                return Float32.(bm)
                            
            elseif sumt + sumv == 0
                info("not found in training/test wav files...passing through")
                return ones(Float32,size(𝕏))
            else
                error("multiple maps of training/test wav files")
            end
        end
        
        
        # reconstruct
        if isempty(model)
            bmr = bm_reference()
            𝕏 .*= bmr
        else
            bmi = bm_inference(TF{Float32}(model), 𝕏, r, nat, μ, σ)
            bmr = bm_reference()
            𝕏 .*= bmi
            bmr .= abs.(bmi.-bmr)
        end
        
        y = STFT2.stft2(𝕏, h, nfft, nhp, STFT2.sqrthann)*2
        WAV.wavwrite(y, wav[1:end-4]*"-processed.wav", Fs=fs)
        
        return bmr
end




function vola_processing(nfft::Int64, nhp::Int64, nat::Int64, ntxt::Int64, μ::Array{Float32,1}, σ::Array{Float32,1}, tid, vid, wav::String, nn::TF{Float32})
    
        r = div(ntxt-1,2)
        x, sr = WAV.wavread(wav)
        x = Float32.(x)        
        𝕏, h = STFT2.stft2(view(x,:,1), nfft, nhp, STFT2.sqrthann)

        function bm_reference()
            
            tbi = contains.(names(tid),basename(wav))
            vbi = contains.(names(vid),basename(wav))
            sumt = sum(tbi)
            sumv = sum(vbi)
            if sumt + sumv == 1
                hit = sumt > sumv ? names(tid)[tbi][1] : names(vid)[vbi][1]
                bm = sumt > sumv ?  read(tid[hit]["bm"]) : read(vid[hit]["bm"])
                return Float32.(bm)
                            
            elseif sumt + sumv == 0
                info("not found in training/test wav files...passing through")
                return ones(Float32,size(𝕏))
            else
                error("multiple maps of training/test wav files")
            end
        end
        
        # guestimate the bm mask and log the error
        bmi = bm_inference(nn, 𝕏, r, nat, μ, σ)
        bmr = bm_reference()
        𝕏 .*= bmi
        bmr .= abs.(bmi.-bmr)

        # reconstruct
        y = STFT2.stft2(𝕏, h, nfft, nhp, STFT2.sqrthann)*2
        WAV.wavwrite(y, wav[1:end-4]*"-processed.wav", Fs=sr)
        
        return bmr
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