module DATA
# utility functions for data manipualtions


import WAV
import SHA

include("ui.jl")




# Deprecated! -> randstring()
# function rand_alphanum(n::Int64)
#     an = collect("0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
#     x = Array{Char,1}(n)
#     for i = 1:n
#         x[i] = rand(an)
#     end
#     String(x)
# end



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
    

# checksum of file list
function checksum(list::Array{String,1})
    
    d = zeros(UInt8, 32)
    n = length(list)
    p = UI.Progress(10)
    
    for (i, j) in enumerate(list)
        d .+= open(j) do f
            SHA.sha256(f)
        end
        UI.update(p, i, n)
    end
    d
end
    

function touch_checksum(path::String)
    d = zeros(UInt8, 32)
    d .+= SHA.sha256("randomly set the checksum of path")
    p = joinpath(path, "index.sha256")
    writedlm(p, d)
    nothing
end
    
    
function update_checksum(path::String)
    
    p = joinpath(path, "index.sha256")
    writedlm(p, checksum(list(path, t = ".wav")))
    info("checksum updated in $p")
    nothing
end

function verify_checksum(path::String)
    
    p = view(readdlm(joinpath(path, "index.sha256"), UInt8), :, 1)
    q = checksum(list(path, t = ".wav"))
    ok = (0x0 == sum(p - q))
end
    
    
    
    
# resample entire folder to another while maintain folder structure
# 1. need ffmpeg installed as backend
# 2. need sox install as resample engine
function resample(path_i::String, path_o::String, target_fs; source_type=".wav", mix_to_mono=false)
    
    a = list(path_i, t = source_type)
    n = length(a)
    u = Array{Int64,1}(n)
    
    name = randstring(rand(4:32))
    tm = joinpath(tempdir(), "$(name).wav")
    
    for (i, j) in enumerate(a)
        run(`ffmpeg -y -i $j $tm`)
        p = joinpath(path_o, relpath(dirname(j), path_i))
        mkpath(p)
        p = joinpath(p, replace(basename(j), source_type, ".wav"))
        run(`sox $tm -r $(target_fs) $p`)
                
        x, fs = WAV.wavread(p)
        assert(fs == typeof(fs)(target_fs))
        if mix_to_mono
            WAV.wavwrite(mean(x,2), p, Fs=fs, nbits=32)
        else
            WAV.wavwrite(x, p, Fs=fs, nbits=32)
        end
        u[i] = size(x, 1)
        println("$i/$n complete")
    end

    rm(tm, force = true)
    println("max: $(maximum(u) / target_fs) seconds")
    println("min: $(minimum(u) / target_fs) seconds")
    nothing
end




function writebin(file::String, data::AbstractArray{T}) where T<:Number
    open(file, "w") do f
        for i in data
            write(f, i)
        end
    end
end


function readbin(file::String, dtype::Type{T}) where T<:Number
    open(file, "r") do f
        reinterpret(dtype, read(f))
    end
end






function deduplicate(path::String, t=".wav")
    a = list(path, t=t)
    redundent::Int64 = 0

    function digest(x)
        open(x,"r") do fid 
            SHA.sha256(fid) 
        end
    end

    # input: array of redundent files
    # side-effect: files removed
    #              number of files removed accumulated
    function remove_redundency(x::Array{String})
        len = [length(list(dirname(i), t=t)) for i in x]
        keep = indmin(len)
        for (i,v) in enumerate(x)
            i != keep && rm(v)
        end
        redundent += (length(x)-1)
        nothing
    end

    open(joinpath(path,"dedup.log"),"w") do f 
        
        println("Building checksum list...")
        chk = [digest(i) for i in a]
        println("Done.")

        n = length(chk)
        hit::Bool = false
        cache = Set{Int64}()

        for i = 1:n-1
            if !in(i, cache)
                hit = false
                dup = String[]
                for j = i+1:n
                    if chk[i]==chk[j]
                        push!(cache,j)
                        write(f,"$(a[j])\n")
                        push!(dup, a[j])
                        hit = true
                    end
                end
                if hit == true
                    write(f,"[+$(a[i])+]\n\n")
                    push!(dup, a[i])
                    remove_redundency(dup)
                end
            end
        end
        println("$redundent files removed")
    end
    nothing
end



# module
end