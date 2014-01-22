module Data.Array.Accelerate.BLAS.Internal.Scal where

import Data.Array.Accelerate.BLAS.Internal.Common

import Data.Array.Accelerate
import Data.Array.Accelerate.CUDA.Foreign
import qualified Foreign.CUDA.BLAS as BL
import Prelude hiding (zipWith, map)

cudaScalF :: (Scalar Float, Vector Float) -> CIO (Vector Float)
cudaScalF (a, x) = do
    let n  =  arraySize (arrayShape x)
    x'     <- allocateArray (arrayShape x)
    copyArray x x'

    xptr   <- devVF x
    x'ptr  <- devVF x'
    aptr   <- devSF a

    liftIO $ BL.withCublas $ \handle -> execute handle n aptr x'ptr

    return x'

    where
        execute h n ap x'p =
            BL.sscal h n ap x'p 1

cudaScalD :: (Scalar Double, Vector Double) -> CIO (Vector Double)
cudaScalD (a, x) = do
    let n  =  arraySize (arrayShape x)
    x'     <- allocateArray (arrayShape x)
    copyArray x x'

    x'ptr  <- devVD x'
    aptr   <- devSD a

    liftIO $ BL.withCublas $ \handle -> execute handle n aptr x'ptr

    return x'

    where
        execute h n ap x'p =
            BL.dscal h n ap x'p 1

sscal :: Acc (Scalar Float) -> Acc (Vector Float) -> Acc (Vector Float)
sscal a x = foreignAcc foreignScalF pureScalF $ lift (a,x)
  where foreignScalF = CUDAForeignAcc "cudaScalF" cudaScalF
        
        pureScalF :: Acc (Scalar Float, Vector Float) -> Acc (Vector Float)
        pureScalF vs = let (a, x) = unlift vs
                           a'     = the a
                       in map (* a') x

dscal :: Acc (Scalar Double) -> Acc (Vector Double) -> Acc (Vector Double)
dscal a x = foreignAcc foreignScalD pureScalD $ lift (a,x)
  where foreignScalD = CUDAForeignAcc "cudaScalD" cudaScalD
        
        pureScalD :: Acc (Scalar Double, Vector Double) -> Acc (Vector Double)
        pureScalD vs = let (a, x) = unlift vs
                           a'     = the a
                       in map (* a') x