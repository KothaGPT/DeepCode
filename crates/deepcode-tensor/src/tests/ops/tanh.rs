#[deepcode_tensor_testgen::testgen(tanh)]
mod tests {
    use super::*;
    use deepcode_tensor::{Tensor, TensorData};
    use deepcode_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn should_support_tanh_ops() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.tanh();
        let expected =
            TensorData::from([[0.0, 0.761594, 0.964028], [0.995055, 0.999329, 0.999909]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
