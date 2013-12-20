{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}

module Data.Array.Accelerate.BLAS.Common 
  ( devVF
  , devVD
  , devSF
  , devSD 
  , allocScalar
  , allocVector ) where

import Data.Array.Accelerate
import Data.Array.Accelerate.CUDA.Foreign
import Foreign.CUDA.Ptr

devVF :: Vector Float -> CIO (DevicePtr Float)
devVF v = devicePtrsOfArray v >>= \((), p) -> return p
{-# INLINE devVF #-}

devVD :: Vector Double -> CIO (DevicePtr Double)
devVD v = devicePtrsOfArray v >>= \((), p) -> return p
{-# INLINE devVD #-}

devSF :: Scalar Float -> CIO (DevicePtr Float)
devSF s = devicePtrsOfArray s >>= \((), p) -> return p
{-# INLINE devSF #-}

devSD :: Scalar Double -> CIO (DevicePtr Double)
devSD s = devicePtrsOfArray s >>= \((), p) -> return p
{-# INLINE devSD #-}

allocScalar :: Elt e => CIO (Scalar e)
allocScalar = allocateArray Z
{-# INLINE allocScalar #-}

allocVector :: Elt e => Int -> CIO (Vector e)
allocVector n = allocateArray (Z :. n)
{-# INLINE allocVector #-}