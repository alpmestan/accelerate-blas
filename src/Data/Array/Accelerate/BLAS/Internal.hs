{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}

module Data.Array.Accelerate.BLAS.Internal
  ( -- * Dot product for various types
    cudaDPF
  ) where

import Prelude hiding (zipWith)
import Control.Applicative ((<$>))
import Data.Array.Accelerate
import qualified Data.Array.Accelerate.Array.Sugar as S
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.CUDA.Foreign
import Foreign.CUDA.BLAS.Helper
import Foreign.CUDA.BLAS.Level1
-- import Foreign.C
import qualified Foreign.CUDA as CUDA

-- TODO: cache the context accross multiple runs with the same dimension!
-- maybe reuse the 'unsafePerformIO' trick in accelerate-fft, see FFT.hs
cudaDotProductF :: (Vector Float, Vector Float) -- ^ vectors we're running dot product on
                -> CIO (Scalar Float)           -- ^ result of the dot product
cudaDotProductF (v1, v2) = do
    let n = arraySize (arrayShape v1)

    -- allocate result scalar
    o <- allocateArray Z

    -- get device pointers on the GPU memory 
    -- for the two vectors and the result
    ((), v1ptr)  <- devicePtrsOfArray v1
    ((), v2ptr)  <- devicePtrsOfArray v2
    ((), outPtr) <- devicePtrsOfArray o

    -- get the CUBLAS context (from my cublas binding)
    handle <- liftIO $ create

    -- run the computation
    liftIO $ execute handle n v1ptr v2ptr outPtr

    -- clean up the CUBLAS context (from my cublas binding)
    liftIO $ destroy handle

    -- return the output array, that now contains the result
    return o

    where
      execute h n v1ptr v2ptr optr
        = sdot h n v1ptr 1 v2ptr 1 optr

-- | Execute the dot product of the two vectors using
--   the CUDA backend if available, fallback to a "pure"
--   implementation otherwise:
--
--   >>> A.fold (+) 0 $ A.zipWith (*) v1 v2
cudaDPF :: Acc (Vector Float) -> Acc (Vector Float) -> Acc (Scalar Float)
cudaDPF v1 v2 = foreignAcc foreignDPF pureDPF $ lift (v1, v2)
  where foreignDPF = CUDAForeignAcc "cudaDotProductF" cudaDotProductF
        
        pureDPF :: Acc (Vector Float, Vector Float) -> Acc (Scalar Float)
        pureDPF vs = let (u, v) = unlift vs
                     in fold (+) 0 $ zipWith (*) u v
