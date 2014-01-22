{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}

module Data.Array.Accelerate.BLAS.Level1
  ( -- * Dot product
    sdot
  , ddot
    -- * Vector scaling and addition
  , saxpy
  , daxpy
    -- * Absolute sum of vector elements
  , sasum
  , dasum
    -- Norm 2
  , snrm2
  , dnrm2
  ) where

-- dot products
import Data.Array.Accelerate.BLAS.Internal.Dot

-- a.x + y
import Data.Array.Accelerate.BLAS.Internal.Axpy

-- absolute sum of elements
import Data.Array.Accelerate.BLAS.Internal.Asum

-- norm 2
import Data.Array.Accelerate.BLAS.Internal.Nrm2