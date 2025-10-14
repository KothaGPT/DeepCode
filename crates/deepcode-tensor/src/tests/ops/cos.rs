#[deepcode_tensor_testgen::testgen(cos)]
mod tests {
    use super::*;
    use deepcode_tensor::{Tensor, TensorData};
    use deepcode_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn should_support_cos_ops() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.cos();
        let expected = TensorData::from([[1.0, 0.54030, -0.41615], [-0.98999, -0.65364, 0.28366]]);

        // Metal has less precise trigonometric functions
        let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, tolerance);
    }
}
