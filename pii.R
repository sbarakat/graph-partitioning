# Installing igraph and pii on MacOS issue:
#This is clearly (to me anyway!) a bug in Xcode.  I traced it to a bogus reference to /usr/lib/system/libsystem_darwin.dylib in /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk/usr/lib/libSystem.B.tbd
#If you edit that file (I made a cp first) and remove the reference to libsystem_darwin.dylib, you should be able to compile lxml just fine.  I also tested a clean build of CPython's git master and didn't have any problems after editing that file, so AFAICT there are no ill side-effects.

args = commandArgs(trailingOnly=TRUE)
if(length(args) == 0) {
  stop("Requires the input graph file path/name.")
}

if (!require('pii')) {
  print('PII package is required, currently missing.')
  print('install.packages("devtools")')
  print('devtools::install_github("jfaganUK/pii") (slow)')
  stopifnot(require('pii'))
}
if (!require('igraph')) {
  print('IGRAPH package is required, currently missing.')
  print('install.packages("igraph")')
  stopifnot(require('pii'))
}
library(pii)
library(igraph)

g <- read_graph(args[1])
result <- pii(g)

print("#magic_code#")
print(result)
print("#magic_end#")
