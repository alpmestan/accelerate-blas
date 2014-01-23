module Data.Array.Accelerate.BLAS.Internal.Asum where

import Data.Array.Accelerate.BLAS.Internal.Common

import Data.Array.Accelerate
import Data.Array.Accelerate.CUDA.Foreign
import qualified Foreign.CUDA.BLAS as BL
import Prelude hiding (zipWith, map)

cudaAsumF :: Vector Float -> CIO (Scalar Float)
cudaAsumF x = do
    let n  =  arraySize (arrayShape x)
    res    <- allocScalar

    xptr   <- devVF x
    resptr <- devSF res

    liftIO $ BL.withCublas $ \handle -> execute handle n xptr resptr

    return res

    where
        execute h n xp rp =
            BL.sasum h n xp 1 rp

cudaAsumD :: Vector Double -> CIO (Scalar Double)
cudaAsumD x = do
    let n  =  arraySize (arrayShape x)
    res    <- allocScalar

    xptr   <- devVD x
    resptr <- devSD res

    liftIO $ BL.withCublas $ \handle -> execute handle n xptr resptr

    return res

    where
        execute h n xp rp =
            BL.dasum h n xp 1 rp

-- | Returns the sum of the absolute value of the `Float`s
--   contained in the vector.
sasum :: Acc (Vector Float) -> Acc (Scalar Float)
sasum = foreignAcc foreignAsumF pureAsumF
  where foreignAsumF = CUDAForeignAcc "cudaAsumF" cudaAsumF
        
        pureAsumF :: Acc (Vector Float) -> Acc (Scalar Float)
        pureAsumF = fold (+) 0 . map abs

-- | Returns the sum of the absolute value of the `Double`s
--   contained in the vector.
dasum :: Acc (Vector Double) -> Acc (Scalar Double)
dasum = foreignAcc foreignAsumD pureAsumD
  where foreignAsumD = CUDAForeignAcc "cudaAsumD" cudaAsumD
        
        pureAsumD :: Acc (Vector Double) -> Acc (Scalar Double)
        pureAsumD = fold (+) 0 . map abs