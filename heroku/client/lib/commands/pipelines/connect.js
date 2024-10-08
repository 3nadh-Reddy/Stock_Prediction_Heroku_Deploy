"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const command_1 = require("@heroku-cli/command");
const core_1 = require("@oclif/core");
const api_1 = require("../../lib/api");
const github_api_1 = require("../../lib/pipelines/github-api");
const kolkrabbi_api_1 = require("../../lib/pipelines/kolkrabbi-api");
const get_github_token_1 = require("../../lib/pipelines/setup/get-github-token");
const get_name_and_repo_1 = require("../../lib/pipelines/setup/get-name-and-repo");
const get_repo_1 = require("../../lib/pipelines/setup/get-repo");
const validate_1 = require("../../lib/pipelines/setup/validate");
class Connect extends command_1.Command {
    async run() {
        const { args, flags } = await this.parse(Connect);
        const combinedInputs = {
            name: args.name,
            repo: flags.repo,
        };
        const errors = (0, validate_1.nameAndRepo)({ repo: flags.repo });
        if (errors.length > 0) {
            this.error(errors.join(', '));
            return;
        }
        const kolkrabbi = new kolkrabbi_api_1.default(this.config.userAgent, () => this.heroku.auth);
        const github = new github_api_1.default(this.config.userAgent, await (0, get_github_token_1.default)(kolkrabbi));
        const { name: pipelineName, repo: repoName, } = await (0, get_name_and_repo_1.default)(combinedInputs);
        const repo = await (0, get_repo_1.default)(github, repoName);
        const pipeline = await (0, api_1.getPipeline)(this.heroku, pipelineName);
        core_1.ux.action.start('Linking to repo');
        await kolkrabbi.createPipelineRepository(pipeline.body.id, repo.id);
        core_1.ux.action.stop();
    }
}
exports.default = Connect;
Connect.description = 'connect a GitHub repo to an existing pipeline';
Connect.examples = [
    '$ heroku pipelines:connect my-pipeline -r githuborg/reponame',
];
Connect.flags = {
    repo: command_1.flags.string({
        name: 'repo',
        char: 'r',
        description: 'the GitHub repository to connect to',
        required: true,
    }),
};
Connect.args = {
    name: core_1.Args.string({
        description: 'name of pipeline',
        required: true,
    }),
};
