module DATA
# utility functions for data manipualtions


import WAV
import SHA


include("ui.jl")



# list all subfolders or files of specified type
# 1. list(path) will list all subfolder names under path, w/o its parent paths!  
# 2. list(path, t=".wav") will list all wav files under path 
function list(path::String; t = "")
    
    x = Array{String,1}()
    for (root, dirs, files) in walkdir(path)
        for dir in dirs
            isempty(t) && push!(x, dir)
        end
        for file in files
            !isempty(t) && lowercase(file[end-length(t)+1:end])==lowercase(t) && push!(x, joinpath(root, file))
        end
    end
    x
end
    
    
function checksum(list::Array{String,1})
    
    d = zeros(UInt8, 32)
    n = length(list)
    p = UI.Progress(10)
    
    for (i, j) in enumerate(list)
        d += open(j) do f
            SHA.sha256(f)
        end
        UI.update(p, i, n)
    end
    d
end
    
    
    
    
function update_checksum(path::String)
    
    p = joinpath(path, "index.sha256")
    writedlm(p, checksum(list(path, t = ".wav")))
    info("checksum updated in $p")
end

function verify_checksum(path::String)
    
    p = view(readdlm(joinpath(path, "index.sha256"), UInt8), :, 1)
    q = checksum(list(path, t = ".wav"))
    ok = (0x0 == sum(p - q))
end
    
    
    
    
# resample entire folder to another while maintain folder structure
# 1. need ffmpeg installed as backend
# 2. need sox install as resample engine
function resample(path_i::String, path_o::String, target_fs)
    
    a = list(path_i, t = ".wav")
    n = length(a)
    u = Array{Int64,1}(n)
    
    tm = joinpath(tempdir(), "a.wav")
    for (i, j) in enumerate(a)

        run(`ffmpeg -y -i $j $tm`)
        p = joinpath(path_o, relpath(dirname(j), path_i))
        mkpath(p)
        p = joinpath(p, basename(j))
        run(`sox $tm -r $(target_fs) $p`)
                
        x, fs = WAV.wavread(p)
        assert(fs == typeof(fs)(target_fs))
        u[i] = size(x, 1)
        info("$i/$n complete")
    end

    info("max: $(maximum(u) / target_fs) seconds")
    info("min: $(minimum(u) / target_fs) seconds")
    rm(tm, force = true)
end


# module
end