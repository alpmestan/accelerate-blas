module Data.Array.Accelerate.BLAS.Internal.Nrm2 where

import Data.Array.Accelerate.BLAS.Internal.Common

import Data.Array.Accelerate
import Data.Array.Accelerate.CUDA.Foreign
import qualified Foreign.CUDA.BLAS as BL
import Prelude hiding (zipWith, map)

cudaNrm2F :: Vector Float -> CIO (Scalar Float)
cudaNrm2F x = do
    let n  =  arraySize (arrayShape x)
    res    <- allocScalar

    xptr   <- devVF x
    resptr <- devSF res

    liftIO $ BL.withCublas $ \handle -> execute handle n xptr resptr

    return res

    where
        execute h n xp rp =
            BL.snrm2 h n xp 1 rp

cudaNrm2D :: Vector Double -> CIO (Scalar Double)
cudaNrm2D x = do
    let n  =  arraySize (arrayShape x)
    res    <- allocScalar

    xptr   <- devVD x
    resptr <- devSD res

    liftIO $ BL.withCublas $ \handle -> execute handle n xptr resptr

    return res

    where
        execute h n xp rp =
            BL.dnrm2 h n xp 1 rp

snrm2 :: Acc (Vector Float) -> Acc (Scalar Float)
snrm2 = foreignAcc foreignNrm2F pureNrm2F
  where foreignNrm2F = CUDAForeignAcc "cudaNrm2F" cudaNrm2F
        
        pureNrm2F :: Acc (Vector Float) -> Acc (Scalar Float)
        pureNrm2F = map sqrt . fold (+) 0 . map (\x -> x*x)

dnrm2 :: Acc (Vector Double) -> Acc (Scalar Double)
dnrm2 = foreignAcc foreignNrm2D pureNrm2D
  where foreignNrm2D = CUDAForeignAcc "cudaNrm2D" cudaNrm2D
        
        pureNrm2D :: Acc (Vector Double) -> Acc (Scalar Double)
        pureNrm2D = map sqrt . fold (+) 0 . map (\x -> x*x)
