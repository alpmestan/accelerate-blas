This is an ongoing work that exposes CUBLAS's features through operations on `Accelerate` arrays.

Install
=======

To check that it works for you, make sure you have the CUDA toolkit, which includes the CUBLAS library. You can see the instructions [here](http://hackage.haskell.org/package/accelerate-cuda).

You also need [hs-cublas](http://github.com/alpmestan/hs-cublas). So once the CUDA toolkit is installed on your system, just do:

``` shell
$ git clone https://github.com/alpmestan/hs-cublas.git
$ git clone https://github.com/alpmestan/accelerate-blas.git
$ cd accelerate-blas
$ cabal sandbox init
$ cabal sandbox add-source ../hs-cublas/
$ cabal install
```

Misc
====

This library doesn't cover all of cublas yet. In addition to covering more, care should be taken to share the handle to the cublas library among calls, because for now we make up a handle for every ffi call.

Also, some of the cublas functions overwrite the vector we give it. We could take advantage of this, but right now I copy, to keep our beloved referential transparency.