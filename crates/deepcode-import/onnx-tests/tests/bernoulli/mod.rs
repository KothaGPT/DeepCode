use crate::include_models;
include_models!(bernoulli);

#[cfg(test)]
mod tests {
    use super::*;
    use deepcode::tensor::Shape;
    use deepcode::tensor::Tensor;

    use crate::backend::TestBackend;

    #[test]
    fn bernoulli() {
        let device = Default::default();
        let model = bernoulli::Model::<TestBackend>::new(&device);

        let shape = Shape::from([10]);
        let input = Tensor::<TestBackend, 1>::full(Shape::from([10]), 0.5, &device);
        let output = model.forward(input);
        assert_eq!(shape, output.shape());
    }
}
