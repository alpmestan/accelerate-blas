{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}

module Data.Array.Accelerate.BLAS.Internal
  ( -- * Dot product for various types
    cudaDPF
  ) where

import Prelude hiding (zipWith)
-- import Control.Applicative ((<$>))
import Data.Array.Accelerate
-- import qualified Data.Array.Accelerate.Array.Sugar as S
-- import Data.Array.Accelerate.Type
import Data.Array.Accelerate.CUDA.Foreign
import Foreign.CUDA.BLAS
-- import Foreign.C
-- import qualified Foreign.CUDA as CUDA

-- TODO: cache the context accross multiple runs with the same dimension!
-- maybe reuse the 'unsafePerformIO' trick in accelerate-fft, see FFT.hs
cudaDotProductF :: (Vector Float, Vector Float) -- ^ vectors we're running dot product on
                -> CIO (Scalar Float)           -- ^ result of the dot product
cudaDotProductF (v1, v2) = do
    let n = arraySize (arrayShape v1)

    -- allocate memory on device for the output (result)
    output <- allocateArray Z

    -- get device pointers on the GPU memory 
    -- for the two vectors and the result
    ((), v1ptr) <- devicePtrsOfArray v1
    ((), v2ptr) <- devicePtrsOfArray v2
    ((), oPtr)  <- devicePtrsOfArray output

    -- get the CUBLAS context
    handle <- liftIO $ cublasCreate

    -- run the computation
    liftIO $ execute handle n v1ptr v2ptr oPtr

    -- clean up the CUBLAS context
    liftIO $ cublasDestroy handle

    -- return the output array, that now contains the result
    return output

    where
      execute h n v1ptr v2ptr oPtr
        = cublasSdot h n v1ptr 1 v2ptr 1 oPtr

-- | Execute the dot product of the two vectors using
--   the CUDA backend if available, fallback to a "pure"
--   implementation otherwise:
--
--   >>> A.fold (+) 0 $ A.zipWith (*) v1 v2
cudaDPF :: Acc (Vector Float) -> Acc (Vector Float) -> Acc (Scalar Float)
cudaDPF v1 v2 = foreignAcc foreignDPF pureDPF $ lift (v1, v2)
  where foreignDPF = CUDAForeignAcc "foreignDotProductF" cudaDotProductF
        
        pureDPF :: Acc (Vector Float, Vector Float) -> Acc (Scalar Float)
        pureDPF vs = let (u, v) = unlift vs
                     in fold (+) 0 $ zipWith (*) u v

{- 
floatingDevicePtr :: Vector e -> CIO (CUDA.DevicePtr e)
floatingDevicePtr v
  = case (floatingType :: FloatingType e) of
        TypeFloat{}   -> singleDevicePtr v
        TypeDouble{}  -> singleDevicePtr v
        TypeCFloat{}  -> CUDA.castDevPtr <$> singleDevicePtr v
        TypeCDouble{} -> CUDA.castDevPtr <$> singleDevicePtr v

singleDevicePtr :: DevicePtrs (S.EltRepr e) ~ ((),CUDA.DevicePtr b) => Vector e -> CIO (CUDA.DevicePtr b)
singleDevicePtr v = Prelude.snd <$> devicePtrsOfArray v
-}