# frozen_string_literal: true

module Micrograd
  class NullOp
    def backward(_anything) = nil
  end

  class AddOp < Struct.new(:x, :y)
    def backward(value)
      x.grad += value.grad
      y.grad += value.grad

      x.backward
      y.backward
    end
  end

  class MulOp < Struct.new(:x, :y)
    def backward(value)
      x.grad += y.data * value.grad
      y.grad += x.data * value.grad

      x.backward
      y.backward
    end
  end

  class TanhOp < Struct.new(:x)
    def backward(value)
      x.grad += 1 - value.data**2
      x.backward
    end
  end

  class PowOp < Struct.new(:x, :n)
    def backward(value)
      x.grad += n * (x ** (n-1)) * value.grad

      x.backward
    end
  end
end
