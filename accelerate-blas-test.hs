{-# LANGUAGE BangPatterns #-}
import Data.Array.Accelerate.CUDA
import Data.Array.Accelerate
import Data.Array.Accelerate.BLAS

import Prelude hiding (fst, snd)

main :: IO ()
main = do
    putStrLn "< (1, .., 10), (0, .., 9) > = "
    print $ run1 f $ (v1, v2)
    putStrLn "< (1, .., 10000), (0, .., 9999) > = "
    print $ run1 g $ (w1, w2)

  where v1, v2 :: Array DIM1 Float
        !v1 = fromList (Z :. 10) [1..10]
        !v2 = fromList (Z :. 10) [0..9]
        w1, w2 :: Array DIM1 Double
        !w1 = fromList (Z :. 10000) [1..10000]
        !w2 = fromList (Z :. 10000) [0..9999]

f :: Acc (Vector Float, Vector Float) -> Acc (Scalar Float)
f vs = let (v1, v2) = (fst vs, snd vs) :: (Acc (Vector Float), Acc (Vector Float))
       in v1 `sDot` v2

g :: Acc (Vector Double, Vector Double) -> Acc (Scalar Double)
g vs = let (v1, v2) = (fst vs, snd vs) :: (Acc (Vector Double), Acc (Vector Double))
       in v1 `dDot` v2