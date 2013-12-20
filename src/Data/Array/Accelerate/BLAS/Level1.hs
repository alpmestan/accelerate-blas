{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}

module Data.Array.Accelerate.BLAS.Level1
  ( -- * Dot product for various types
    sDot
  , dDot
  ) where

import Data.Array.Accelerate.BLAS.Common

import Prelude hiding (zipWith)
import Control.Applicative ((<$>))
import Data.Array.Accelerate
import Data.Array.Accelerate.CUDA.Foreign
import Foreign.CUDA.BLAS.Helper
import Foreign.CUDA.BLAS.Level1

-- TODO: cache the context accross multiple runs with the same dimension!
-- maybe reuse the 'unsafePerformIO' trick in accelerate-fft, see FFT.hs
cudaDotProductF :: (Vector Float, Vector Float) -- ^ vectors we're running dot product on
                -> CIO (Scalar Float)           -- ^ result of the dot product
cudaDotProductF (v1, v2) = do
    let n = arraySize (arrayShape v1)

    -- allocate result scalar
    o <- allocScalar

    -- get device pointers on the GPU memory 
    -- for the two vectors and the result
    -- see Data.Array.Accelerate.BLAS.Common
    v1ptr  <- devVF v1
    v2ptr  <- devVF v2
    outPtr <- devSF o

    -- TODO: avoid doing the init/setPointerMode/cleanup on every call
    --       see the unsafePerformIO trick in accelerate-fft
    liftIO $ withCublas $ \handle -> execute handle n v1ptr v2ptr outPtr

    -- return the output array, that now contains the result
    return o

    where
      execute h n v1ptr v2ptr optr
        = sdot h n v1ptr 1 v2ptr 1 optr

cudaDotProductD :: (Vector Double, Vector Double)
                -> CIO (Scalar Double)
cudaDotProductD (v1, v2) = do
    let n = arraySize (arrayShape v1)
    o <- allocScalar
    v1ptr  <- devVD v1
    v2ptr  <- devVD v2
    outPtr <- devSD o
    liftIO $ withCublas $ \handle -> execute handle n v1ptr v2ptr outPtr
    return o

    where
      execute h n v1ptr v2ptr optr
        = ddot h n v1ptr 1 v2ptr 1 optr    

-- | Execute the dot product of the two vectors using
--   CUBLAS in the CUDA backend if available, fallback to a "pure"
--   implementation otherwise:
--
--   >>> A.fold (+) 0 $ A.zipWith (*) v1 v2
sDot :: Acc (Vector Float) -> Acc (Vector Float) -> Acc (Scalar Float)
sDot v1 v2 = foreignAcc foreignDPF pureDPF $ lift (v1, v2)
  where foreignDPF = CUDAForeignAcc "cudaDotProductF" cudaDotProductF
        
        pureDPF :: Acc (Vector Float, Vector Float) -> Acc (Scalar Float)
        pureDPF vs = let (u, v) = unlift vs
                     in fold (+) 0 $ zipWith (*) u v

-- | Execute the dot product of the two vectors using
--   CUBLAS in the CUDA backend if available, fallback to a "pure"
--   implementation otherwise:
--
--   >>> A.fold (+) 0 $ A.zipWith (*) v1 v2
dDot :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Scalar Double)
dDot v1 v2 = foreignAcc foreignDPD pureDPD $ lift (v1, v2)
  where foreignDPD = CUDAForeignAcc "cudaDotProductD" cudaDotProductD
        
        pureDPD :: Acc (Vector Double, Vector Double) -> Acc (Scalar Double)
        pureDPD vs = let (u, v) = unlift vs
                     in fold (+) 0 $ zipWith (*) u v