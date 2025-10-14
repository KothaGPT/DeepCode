use tracel_xtask::{
    prelude::{clap::ValueEnum, *},
    utils::{
        process::{ExitSignal, ProcessExitError},
        workspace::WorkspaceMember,
    },
};

use crate::NO_STD_CRATES;

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

#[macros::extend_command_args(TestCmdArgs, Target, TestSubCommand)]
pub struct DeepcodeTestCmdArgs {
    /// Test in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: CiTestType,
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, ValueEnum, PartialEq)]
pub enum CiTestType {
    GithubRunner,
    GithubMacRunner,
    GcpCudaRunner,
    GcpVulkanRunner,
    GcpWgpuRunner,
}

fn handle_wgpu_test(member: &str, args: &TestCmdArgs) -> anyhow::Result<()> {
    #[cfg(unix)]
    let filter_err = |e: &&ProcessExitError| {
        e.status.signal() == Some(11) || matches!(e.signal, Some(ExitSignal { code: 11, .. }))
    };
    #[cfg(not(unix))]
    let filter_err = |e: &&ProcessExitError| matches!(e.signal, Some(ExitSignal { code: 11, .. }));

    let workspace_member = WorkspaceMember {
        name: member.into(),
        path: "".into(), // unused
    };

    if let Err(err) = base_commands::test::run_unit_test(&workspace_member, args) {
        let should_ignore = err
            .downcast_ref::<ProcessExitError>()
            .filter(filter_err)
            // Failed to execute unit test for '{member}'
            .map(|e| e.message.contains(member))
            .unwrap_or(false);

        if should_ignore {
            // Ignore intermittent successful failures
            // https://github.com/gfx-rs/wgpu/issues/2949
            // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/4391
            eprintln!("⚠️ Ignored SIGSEGV in wgpu test");
        } else {
            return Err(err);
        }
    }
    Ok(())
}

pub(crate) fn handle_command(
    mut args: DeepcodeTestCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    match context {
        Context::NoStd => {
            ["Default"].iter().try_for_each(|test_target| {
                let mut test_args = vec!["--no-default-features"];
                if *test_target != "Default" {
                    test_args.extend(vec!["--target", *test_target]);
                }
                helpers::custom_crates_tests(
                    NO_STD_CRATES.to_vec(),
                    handle_test_args(&test_args, args.release),
                    None,
                    None,
                    "no-std",
                )
            })?;
            Ok(())
        }
        Context::Std => {
            // 1) Tests with default features
            // ------------------------------
            match args.ci {
                CiTestType::GithubRunner => {
                    // Exclude crates that are not supported on CI
                    args.exclude.extend(vec![
                        "deepcode-cpu".to_string(),
                        "deepcode-cuda".to_string(),
                        "deepcode-rocm".to_string(),
                        // "deepcode-router" uses "deepcode-wgpu" for the tests.
                        "deepcode-router".to_string(),
                        "deepcode-tch".to_string(),
                        "deepcode-wgpu".to_string(),
                    ]);

                    // Deepcode remote tests don't work on windows for now
                    #[cfg(target_os = "windows")]
                    {
                        args.exclude.extend(vec!["deepcode-remote".to_string()]);
                    };

                    base_commands::test::handle_command(
                        args.clone().try_into().unwrap(),
                        env,
                        context,
                    )?;
                }
                CiTestType::GithubMacRunner => {
                    args.target = Target::AllPackages;
                    args.only.push("deepcode-wgpu".to_string());
                    args.features
                        .get_or_insert_with(Vec::new)
                        .push("metal".to_string());

                    base_commands::test::handle_command(
                        args.clone().try_into().unwrap(),
                        env,
                        context,
                    )?;
                }
                CiTestType::GcpCudaRunner => {
                    args.target = Target::AllPackages;
                    args.only.push("deepcode-cuda".to_string());

                    base_commands::test::handle_command(
                        args.clone().try_into().unwrap(),
                        env,
                        context,
                    )?;
                }
                CiTestType::GcpVulkanRunner => {
                    let mut args_vulkan = args.clone();
                    args_vulkan
                        .features
                        .get_or_insert_with(Vec::new)
                        .push("vulkan".to_string());

                    let mut args_vulkan = args_vulkan.clone().try_into().unwrap();
                    handle_wgpu_test("deepcode-wgpu", &args_vulkan)?;

                    args_vulkan.features = Some(vec!["test-vulkan".into()]);
                    handle_wgpu_test("deepcode-core", &args_vulkan)?;
                    handle_wgpu_test("deepcode-vision", &args_vulkan)?;
                }
                CiTestType::GcpWgpuRunner => {
                    args.target = Target::AllPackages;
                    // "deepcode-router" uses "deepcode-wgpu" for the tests.
                    let mut args_wgpu = args.clone().try_into().unwrap();
                    handle_wgpu_test("deepcode-wgpu", &args_wgpu)?;
                    handle_wgpu_test("deepcode-router", &args_wgpu)?;

                    args_wgpu.features = Some(vec!["test-wgpu".into()]);
                    handle_wgpu_test("deepcode-core", &args_wgpu)?;
                    handle_wgpu_test("deepcode-vision", &args_wgpu)?;
                }
            }

            // 2) Specific additional commands to test specific features
            // ---------------------------------------------------------
            match args.ci {
                CiTestType::GithubRunner => {
                    // deepcode-dataset
                    helpers::custom_crates_tests(
                        vec!["deepcode-dataset"],
                        handle_test_args(&["--all-features"], args.release),
                        None,
                        None,
                        "std all features",
                    )?;

                    // deepcode-core
                    helpers::custom_crates_tests(
                        vec!["deepcode-core"],
                        handle_test_args(
                            &["--features", "test-tch,record-item-custom-serde"],
                            args.release,
                        ),
                        None,
                        None,
                        "std with features: test-tch,record-item-custom-serde",
                    )?;

                    // deepcode-vision
                    helpers::custom_crates_tests(
                        vec!["deepcode-vision"],
                        handle_test_args(&["--features", "test-cpu"], args.release),
                        None,
                        None,
                        "std cpu",
                    )?;
                }
                CiTestType::GcpCudaRunner => (),
                CiTestType::GcpVulkanRunner | CiTestType::GcpWgpuRunner => (), // handled in tests above
                CiTestType::GithubMacRunner => {
                    // deepcode-candle
                    helpers::custom_crates_tests(
                        vec!["deepcode-candle"],
                        handle_test_args(&["--features", "accelerate"], args.release),
                        None,
                        None,
                        "std accelerate",
                    )?;
                    // deepcode-ndarray
                    helpers::custom_crates_tests(
                        vec!["deepcode-ndarray"],
                        handle_test_args(&["--features", "blas-accelerate"], args.release),
                        None,
                        None,
                        "std blas-accelerate",
                    )?;
                    helpers::custom_crates_tests(
                        vec!["deepcode-core"],
                        handle_test_args(&["--features", "test-metal"], args.release),
                        None,
                        None,
                        "std metal",
                    )?;
                    helpers::custom_crates_tests(
                        vec!["deepcode-vision"],
                        handle_test_args(&["--features", "test-metal"], args.release),
                        None,
                        None,
                        "std metal",
                    )?;
                }
            }
            Ok(())
        }
        Context::All => Context::value_variants()
            .iter()
            .filter(|ctx| **ctx != Context::All)
            .try_for_each(|ctx| {
                handle_command(
                    DeepcodeTestCmdArgs {
                        command: args.command.clone(),
                        target: args.target.clone(),
                        exclude: args.exclude.clone(),
                        only: args.only.clone(),
                        threads: args.threads,
                        jobs: args.jobs,
                        ci: args.ci.clone(),
                        features: args.features.clone(),
                        no_default_features: args.no_default_features,
                        release: args.release,
                        test: args.test.clone(),
                        force: args.force,
                        no_capture: args.no_capture,
                    },
                    env.clone(),
                    ctx.clone(),
                )
            }),
    }
}

fn handle_test_args<'a>(args: &'a [&'a str], release: bool) -> Vec<&'a str> {
    let mut args = args.to_vec();
    if release {
        args.push("--release");
    }
    args
}
