{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}

module Data.Array.Accelerate.BLAS.Level1
  ( -- * Dot product for various types
    sdot
  , ddot
    -- * Vector scaling and addition
  , saxpy
  , daxpy
  ) where

-- dot products
import Data.Array.Accelerate.BLAS.Internal.Dot

-- a.x + y
import Data.Array.Accelerate.BLAS.Internal.Axpy