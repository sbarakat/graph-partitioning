args = commandArgs(trailingOnly=TRUE)
if(length(args) == 0) {
  stop("Requires the input graph file path/name.")
}

if (!require('centiserve')) {
  print('centiserve package is required, currently missing.')
  print('install.packages("centiserve", repos="http://cran.rstudio.com")')
  stopifnot(require('centiserve'))
}
if (!require('igraph')) {
  print('IGRAPH package is required, currently missing.')
  print('install.packages("igraph")')
  stopifnot(require('igraph'))
}

library(igraph)
library(centiserve)

g <- read_graph(args[1])
result <- bottleneck(g)

print("#magic_code#")
print(result)
print("#magic_end#")
