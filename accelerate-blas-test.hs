{-# LANGUAGE BangPatterns #-}
import Data.Array.Accelerate.CUDA
import Data.Array.Accelerate
import Data.Array.Accelerate.BLAS

import Prelude hiding (fst, snd)

main :: IO ()
main = print $ run1 f (v1, v2)
  where v1, v2 :: Array DIM1 Float
        !v1 = fromList (Z :. 10) [1..10]
        !v2 = fromList (Z :. 10) [1..10]

f :: Acc (Vector Float, Vector Float) -> Acc (Scalar Float)
f vs = let (v1, v2) = (fst vs, snd vs) :: (Acc (Vector Float), Acc (Vector Float))
       in cudaDPF v1 v2