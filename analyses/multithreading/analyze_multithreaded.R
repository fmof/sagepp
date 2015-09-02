> data = read.csv('multithreaded_diffs.csv',header=T,sep=',')
> data$ndiff = data$pend - data$pstart  C-c C-c
> data = read.csv('multithreaded_diffs.csv',header=T,sep=',')
> data$pstart = as.numeric(as.POSIXlt(data$start, format = "%H:%M:%OS"))
> data$pend = as.numeric(as.POSIXlt(data$end, format = "%H:%M:%OS"))
> data
   nthreads opt topic batch           start             end     pstart
1         2   0     0     1 15:03:09.576967 15:04:00.922488 1428433390
2         2   0     1     1 15:03:09.577439 15:04:02.596146 1428433390
3         1   0     0     1 15:11:02.375716 15:11:55.997396 1428433862
4         1   0     1     1 15:11:55.997604 15:12:52.603540 1428433916
5         1   3     0     1 15:18:45.275570 15:19:13.356186 1428434325
6         1   3     1     1 15:19:13.356397 15:19:41.995924 1428434353
7         1   3     0     2 15:19:42.553308 15:20:07.854899 1428434383
8         1   3     1     2 15:20:07.855107 15:20:34.882077 1428434408
9         2   3     0     1 15:23:12.591160 15:23:46.158467 1428434593
10        2   3     1     1 15:23:12.594152 15:23:48.277447 1428434593
11        2   3     0     2 15:23:48.874977 15:24:22.724106 1428434629
12        2   3     1     2 15:23:48.875060 15:24:22.734255 1428434629
         pend
1  1428433441
2  1428433443
3  1428433916
4  1428433973
5  1428434353
6  1428434382
7  1428434408
8  1428434435
9  1428434626
10 1428434628
11 1428434663
12 1428434663
> start.agg = aggregate(pstart ~ nthreads + opt + topic, data, function(x) { min(x) } )
> end.agg = aggregate(pend ~ nthreads + opt + topic, data, function(x) { max(x) } )
> joined.agg = merge(start.agg, end.agg)
> joined.agg
  nthreads opt topic     pstart       pend
1        1   0     0 1428433862 1428433916
2        1   0     1 1428433916 1428433973
3        1   3     0 1428434325 1428434408
4        1   3     1 1428434353 1428434435
5        2   0     0 1428433390 1428433441
6        2   0     1 1428433390 1428433443
7        2   3     0 1428434593 1428434663
8        2   3     1 1428434593 1428434663
> start.agg = aggregate(pstart ~ nthreads + opt, data, function(x) { min(x) } )
> end.agg = aggregate(pend ~ nthreads + opt, data, function(x) { max(x) } )
> joined.agg
  nthreads opt topic     pstart       pend
1        1   0     0 1428433862 1428433916
2        1   0     1 1428433916 1428433973
3        1   3     0 1428434325 1428434408
4        1   3     1 1428434353 1428434435
5        2   0     0 1428433390 1428433441
6        2   0     1 1428433390 1428433443
7        2   3     0 1428434593 1428434663
8        2   3     1 1428434593 1428434663
> joined.agg = merge(start.agg, end.agg)
> joined.agg
  nthreads opt     pstart       pend
1        1   0 1428433862 1428433973
2        1   3 1428434325 1428434435
3        2   0 1428433390 1428433443
4        2   3 1428434593 1428434663
> end.agg = aggregate(pend ~ nthreads + opt + batch, data, function(x) { max(x) } )
> start.agg = aggregate(pstart ~ nthreads + opt + batch, data, function(x) { min(x) } )
> joined.agg = merge(start.agg, end.agg)
> joined.agg
  nthreads opt batch     pstart       pend
1        1   0     1 1428433862 1428433973
2        1   3     1 1428434325 1428434382
3        1   3     2 1428434383 1428434435
4        2   0     1 1428433390 1428433443
5        2   3     1 1428434593 1428434628
6        2   3     2 1428434629 1428434663
> joined.agg$ndiff = joined.agg$pend - joined.agg$pstart
> joined.agg
  nthreads opt batch     pstart       pend     ndiff
1        1   0     1 1428433862 1428433973 110.22782
2        1   3     1 1428434325 1428434382  56.72035
3        1   3     2 1428434383 1428434435  52.32877
4        2   0     1 1428433390 1428433443  53.01918
5        2   3     1 1428434593 1428434628  35.68629
6        2   3     2 1428434629 1428434663  33.85928
> mean.agged = aggregate(ndiff ~ nthreads + opt, joined.agg, mean)
> ggplot(mean.agged) + geom_bar(aes(factor(nthreads), ndiff, fill = factor(opt)), stat="identity", position = "dodge")  + theme_bw()
> ggplot(mean.agged) + geom_bar(aes(factor(opt), ndiff, fill = factor(nthreads)), stat="identity", position = "dodge")  + theme_bw()
> ggplot(mean.agged) + geom_bar(aes(factor(opt), ndiff, fill = factor(nthreads)), stat="identity", position = "dodge")  + theme_bw() + scale_y_log10()
> ggplot(mean.agged) + geom_bar(aes(factor(opt), ndiff, fill = factor(nthreads)), stat="identity", position = "dodge")  + theme_bw()
> ggplot(mean.agged) + geom_bar(aes(factor(opt), ndiff, fill = factor(nthreads)), stat="identity", position = "dodge")  + theme_bw() + xlab("Optimization level") + ylab("Time per m-step")
> mean.agged
  nthreads opt     ndiff
1        1   0 110.22782
2        2   0  53.01918
3        1   3  54.52456
4        2   3  34.77278
> div.agged = aggregate(ndiff ~ opts, mean.agged, function(x) { max(x) / min(x) })
Error in model.frame.default(formula = ndiff ~ opts, data = mean.agged) : 
  invalid type (closure) for variable 'opts'
> div.agged = aggregate(ndiff ~ opt, mean.agged, function(x) { max(x) / min(x) })
> div.agged
  opt    ndiff
1   0 2.079018
2   3 1.568024
> 110.22782 / 53.01918
[1] 2.079018
> ggplot(div.agged) + geom_bar(aes(factor(opt), ndiff), stat="identity")  + theme_bw() + xlab("Optimization level") + ylab("Time per m-step")
> ggplot(div.agged) + geom_bar(aes(factor(opt), ndiff), stat="identity")  + theme_bw() + xlab("Optimization level") + ylab("Multithreaded speedup per m-step")
> 
