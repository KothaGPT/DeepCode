pub mod cube_comment {
    use crate::ir::NonSemantic;
    use deepcl_ir::Scope;

    pub fn expand(scope: &mut Scope, content: &str) {
        scope.register(NonSemantic::Comment {
            content: content.to_string(),
        });
    }
}
