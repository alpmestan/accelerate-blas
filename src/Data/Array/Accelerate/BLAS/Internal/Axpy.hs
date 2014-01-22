module Data.Array.Accelerate.BLAS.Internal.Axpy where

import Data.Array.Accelerate.BLAS.Internal.Common

import Data.Array.Accelerate
import Data.Array.Accelerate.CUDA.Foreign
import qualified Foreign.CUDA.BLAS as BL
import Prelude hiding (zipWith, map)

cudaAxpyF :: (Scalar Float, Vector Float, Vector Float)
          -> CIO (Vector Float)
cudaAxpyF (alpha, x, y) = do
    let n = arraySize (arrayShape y)
    y'    <- allocateArray (arrayShape y)
    copyArray y y'
    aptr  <- devSF alpha
    xptr  <- devVF x
    y'ptr <- devVF y'

    liftIO $ BL.withCublas $ \handle -> execute handle n aptr xptr y'ptr

    return y'

    where
        execute h n a xp yp =
            BL.saxpy h n a xp 1 yp 1

cudaAxpyD :: (Scalar Double, Vector Double, Vector Double)
          -> CIO (Vector Double)
cudaAxpyD (alpha, x, y) = do
    let n = arraySize (arrayShape y)
    y'    <- allocateArray (arrayShape y)
    copyArray y y'
    aptr  <- devSD alpha
    xptr  <- devVD x
    y'ptr <- devVD y'

    liftIO $ BL.withCublas $ \handle -> execute handle n aptr xptr y'ptr

    return y'

    where
        execute h n a xp yp =
            BL.daxpy h n a xp 1 yp 1

-- | Execute /alpha.x + y/ where /x, y/ are /vectors/ and /alpha/ is /scalar/, using
--   CUBLAS in the CUDA backend if available, fallback to a "pure"
--   implementation otherwise:
--
--   >>> zipWith (+) y $ map (*alpha) x
saxpy :: Acc (Scalar Float) -> Acc (Vector Float) -> Acc (Vector Float) -> Acc (Vector Float)
saxpy alpha x y = foreignAcc foreignSaxpy pureSaxpy $ lift (alpha, x, y)
  where foreignSaxpy = CUDAForeignAcc "cudaAxpyF" cudaAxpyF
        
        pureSaxpy :: Acc (Scalar Float, Vector Float, Vector Float) -> Acc (Vector Float)
        pureSaxpy vs = let (a, u, v) = unlift vs
                       in zipWith (+) v $ map (*(the a)) u

-- | Execute /alpha.x + y/ using
--   CUBLAS in the CUDA backend if available, fallback to a "pure"
--   implementation otherwise:
--
--   >>> zipWith (+) y $ map (*alpha) x
daxpy :: Acc (Scalar Double) -> Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
daxpy alpha x y = foreignAcc foreignDaxpy pureDaxpy $ lift (alpha, x, y)
  where foreignDaxpy = CUDAForeignAcc "cudaAxpyD" cudaAxpyD
        
        pureDaxpy :: Acc (Scalar Double, Vector Double, Vector Double) -> Acc (Vector Double)
        pureDaxpy vs = let (a, u, v) = unlift vs
                       in zipWith (+) v $ map (* (the a)) u