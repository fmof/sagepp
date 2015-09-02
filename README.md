# SAGE++ --- A C++ SAGE Implementation

Version 1.0.0

This is a C++ 11 implementation of SAGE (Eisenstein et al., 2011).
It has been developed and tested on Linux x86_64, under G++ >= 4.8.4.
It should also compile on a Mac, though a couple Makefile changes may need to be made.

## Dependencies

This relies on:

* Boost >= 1.56
* GSL >= 1.16
* [libLBFGS](http://www.chokkan.org/software/liblbfgs/) v1.10
* [GTest](https://github.com/google/googletest) v1.7.0

Only GTest is included in the repo.

## Building

### Local Configurations
Depending on where you installed the dependencies (except GTest), you may need to update `Makefile.config`.
* If the headers for Boost, GSL and libLBFGS are installed in `/usr/loca/include`, and the shared objects are installed in `/usr/local/lib`, then you do _not_ have to change anything.
* If the headers (shared objects) for Boost, GSL and libLBFGS are installed in the same place (but not `/usr/local/{include,lib}`), then change lines 5 and 6 of `Makefile.config`.
* If the headers (shared objects) for Boost, GSL and libLBFGS are installed in separate directories, then change lines 11, 12, 17, 18, 23, and 24, as appropriate.

### Building the targets

`make help` will display all known targets.
To build the SAGE++ driver, run `make isage test models/sage_driver`.

There are a number of Makefile ENV variables that can be set to change compilation.
Some major ones are:
* `DEBUG={0,1}`: turn on debug compilation (`-g -O0`). Default: 0
* `LINK_HOW={dynamic,static}`: use dynamic or static linking for the `isage` library. Default: dynamic
* `CXX_ISAGE_LOG`: This is meant to change what logging is used. Set to empty to use BOOST logging, but as a WARNING, BOOST should only be used in single-threaded applications! (Default: `-DISAGE_LOG_AS_COUT`)

## Data

Due to licensing issues, I am not releasing the data files.
However, the expected data format is in an "SVM-light" feature format:
* each instance is on its own line
* each line is single-space separated (' ')
* the first column is an integer label, corresponding to the gold label of the instance
* for a document with `X` word-type occurrences, the next `X` columns are of the form `word_id:count`
* all remaining columns are comments, beginning with '#'

For example:

```
1 1:3 2:1 3:2 4:2 5:2 6:2 7:3 8:2 9:1 # doc_1
```

indicates that this (toy) document has label "1", has 18 tokens and 9 word types.

## Licensing

This code is released under GPL v3.0.
Please contact me (ferraro [at] cs [dot] jhu [dot] edu) with any questions.
