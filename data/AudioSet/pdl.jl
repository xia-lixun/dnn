function log_recur(file, r)
    open(file, "a") do fid
        write(fid, r[1] * " " * r[2] * " " * r[3] * " \"" * r[4] * "\"\n")
    end        
end

function log_except(digest, r)
    open("exception.log", "a") do fid
        write(fid, "===\n")
        write(fid, r[1] * " " * r[2] * " " * r[3] * " \"" * r[4] * "\"\n")
        write(fid, digest * "\n")
    end        
end


function pdl(record)

    # pid = myid()
    # wid = workers()
    # wid = sort(wid)

    # channel = find(x->x==pid, wid)[1]
    # assert(in(channel,collect(1:8)))
    # s = score(file, channel)

    ytid = record[1][1:end-1]
    # t0 = parse(Float64, record[2][1:end-1])
    # t1 = parse(Float64, record[3][1:end-1])
    # label = record[4]


    trymax = 1
    isok = false
    try
        while trymax > 0
            dig = readstring(`youtube-dl -f bestaudio --id "https://www.youtube.com/watch?v=$(ytid)"`)
            if ismatch(Regex("100%"), dig)
                log_recur("good.csv", record)
                trymax = 0
                isok = true
            elseif ismatch(Regex("ERROR:"), dig)
                trymax -= 1
            else
                log_except(dig, record)
                trymax -= 1
            end
        end
    catch
        log_except("unavailable", record)
    end

    isok || log_recur("bad.csv", record)
    isok && isfile("$(ytid).webm") && mv("$(ytid).webm", "/media/coc/Data/webm/$(ytid).webm")
    isok && isfile("$(ytid).m4a") && mv("$(ytid).m4a", "/media/coc/Data/m4a/$(ytid).m4a")
    nothing
end
