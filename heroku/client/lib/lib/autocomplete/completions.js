"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CompletionLookup = exports.CompletionMapping = exports.TeamCompletion = exports.StageCompletion = exports.StackCompletion = exports.SpaceCompletion = exports.ScopeCompletion = exports.RoleCompletion = exports.RemoteCompletion = exports.RegionCompletion = exports.ProtocolCompletion = exports.ProcessTypeCompletion = exports.PipelineCompletion = exports.FileCompletion = exports.DynoSizeCompletion = exports.BuildpackCompletion = exports.AppDynoCompletion = exports.AppAddonCompletion = exports.AppCompletion = exports.herokuGet = exports.oneDay = void 0;
const command_1 = require("@heroku-cli/command");
const deps_1 = require("@heroku-cli/command/lib/deps");
const git_1 = require("@heroku-cli/command/lib/git");
const path = require("path");
const lodash_1 = require("lodash");
exports.oneDay = 60 * 60 * 24;
const herokuGet = async (resource, ctx) => {
    const heroku = new command_1.APIClient(ctx.config);
    let { body } = await heroku.get(`/${resource}`, { retryAuth: false });
    if (typeof body === 'string')
        body = JSON.parse(body);
    return body.map((a) => a.name).sort();
};
exports.herokuGet = herokuGet;
exports.AppCompletion = {
    cacheDuration: exports.oneDay,
    options: async (ctx) => {
        const teams = await (0, exports.herokuGet)('teams', ctx);
        const apps = {
            personal: await (0, exports.herokuGet)('users/~/apps', ctx),
            teams: (0, lodash_1.flatten)(await Promise.all(teams.map((team) => (0, exports.herokuGet)(`teams/${team}/apps`, ctx)))),
        };
        return apps.personal.concat(apps.teams);
    },
};
exports.AppAddonCompletion = {
    cacheDuration: exports.oneDay,
    cacheKey: async (ctx) => {
        return ctx.flags && ctx.flags.app ? `${ctx.flags.app}_addons` : '';
    },
    options: async (ctx) => {
        const addons = ctx.flags && ctx.flags.app ? await (0, exports.herokuGet)(`apps/${ctx.flags.app}/addons`, ctx) : [];
        return addons;
    },
};
exports.AppDynoCompletion = {
    cacheDuration: exports.oneDay,
    cacheKey: async (ctx) => {
        return ctx.flags && ctx.flags.app ? `${ctx.flags.app}_dynos` : '';
    },
    options: async (ctx) => {
        const dynos = ctx.flags && ctx.flags.app ? await (0, exports.herokuGet)(`apps/${ctx.flags.app}/dynos`, ctx) : [];
        return dynos;
    },
};
exports.BuildpackCompletion = {
    skipCache: true,
    options: async () => {
        return [
            'heroku/ruby',
            'heroku/nodejs',
            'heroku/clojure',
            'heroku/python',
            'heroku/java',
            'heroku/gradle',
            'heroku/scala',
            'heroku/php',
            'heroku/go',
        ];
    },
};
const ConfigCompletion = {
    cacheDuration: 60 * 60 * 24 * 7,
    cacheKey: async (ctx) => {
        return ctx.flags && ctx.flags.app ? `${ctx.flags.app}_config_vars` : '';
    },
    options: async (ctx) => {
        const heroku = new command_1.APIClient(ctx.config);
        if (ctx.flags && ctx.flags.app) {
            const { body: configs } = await heroku.get(`/apps/${ctx.flags.app}/config-vars`, { retryAuth: false });
            return Object.keys(configs);
        }
        return [];
    },
};
const ConfigSetCompletion = {
    cacheDuration: 60 * 60 * 24 * 7,
    cacheKey: async (ctx) => {
        return ctx.flags && ctx.flags.app ? `${ctx.flags.app}_config_set_vars` : '';
    },
    options: async (ctx) => {
        const heroku = new command_1.APIClient(ctx.config);
        if (ctx.flags && ctx.flags.app) {
            const { body: configs } = await heroku.get(`/apps/${ctx.flags.app}/config-vars`, { retryAuth: false });
            return Object.keys(configs).map(k => `${k}=`);
        }
        return [];
    },
};
exports.DynoSizeCompletion = {
    cacheDuration: exports.oneDay * 90,
    options: async (ctx) => {
        let sizes = await (0, exports.herokuGet)('dyno-sizes', ctx);
        if (sizes)
            sizes = sizes.map(s => s.toLowerCase());
        return sizes;
    },
};
exports.FileCompletion = {
    skipCache: true,
    options: async () => {
        const files = await deps_1.deps.file.readdir(process.cwd());
        return files;
    },
};
exports.PipelineCompletion = {
    cacheDuration: exports.oneDay,
    options: async (ctx) => {
        const pipelines = await (0, exports.herokuGet)('pipelines', ctx);
        return pipelines;
    },
};
exports.ProcessTypeCompletion = {
    skipCache: true,
    options: async () => {
        let types = [];
        const procfile = path.join(process.cwd(), 'Procfile');
        try {
            const buff = await deps_1.deps.file.readFile(procfile);
            types = buff
                .toString()
                .split('\n')
                .map(s => {
                if (!s)
                    return false;
                const m = s.match(/^([A-Za-z0-9_-]+)/);
                return m ? m[0] : false;
            })
                .filter(t => t);
        }
        catch (error) {
            if (error.code !== 'ENOENT')
                throw error;
        }
        return types;
    },
};
exports.ProtocolCompletion = {
    cacheDuration: 60 * 60 * 24 * 365, options: async () => {
        return ['tcp', 'udp', 'icmp', '0-255', 'any'];
    },
};
exports.RegionCompletion = {
    cacheDuration: exports.oneDay * 7,
    options: async (ctx) => {
        const regions = await (0, exports.herokuGet)('regions', ctx);
        return regions;
    },
};
exports.RemoteCompletion = {
    skipCache: true,
    options: async () => {
        const remotes = (0, git_1.getGitRemotes)((0, git_1.configRemote)());
        return remotes.map(r => r.remote);
    },
};
exports.RoleCompletion = {
    skipCache: true,
    options: async () => {
        return ['admin', 'collaborator', 'member', 'owner'];
    },
};
exports.ScopeCompletion = {
    skipCache: true,
    options: async () => {
        return ['global', 'identity', 'read', 'write', 'read-protected', 'write-protected'];
    },
};
exports.SpaceCompletion = {
    cacheDuration: exports.oneDay,
    options: async (ctx) => {
        const spaces = await (0, exports.herokuGet)('spaces', ctx);
        return spaces;
    },
};
exports.StackCompletion = {
    cacheDuration: exports.oneDay,
    options: async (ctx) => {
        const stacks = await (0, exports.herokuGet)('stacks', ctx);
        return stacks;
    },
};
exports.StageCompletion = {
    skipCache: true,
    options: async () => {
        return ['test', 'review', 'development', 'staging', 'production'];
    },
};
exports.TeamCompletion = {
    cacheDuration: exports.oneDay,
    options: async (ctx) => {
        const teams = await (0, exports.herokuGet)('teams', ctx);
        return teams;
    },
};
exports.CompletionMapping = {
    app: exports.AppCompletion,
    addon: exports.AppAddonCompletion,
    dyno: exports.AppDynoCompletion,
    buildpack: exports.BuildpackCompletion,
    config: ConfigCompletion,
    configSet: ConfigSetCompletion,
    dynosize: exports.DynoSizeCompletion,
    pipeline: exports.PipelineCompletion,
    processtype: exports.ProcessTypeCompletion,
    region: exports.RegionCompletion,
    remote: exports.RemoteCompletion,
    role: exports.RoleCompletion,
    scope: exports.ScopeCompletion,
    space: exports.SpaceCompletion,
    stack: exports.StackCompletion,
    stage: exports.StageCompletion,
    team: exports.TeamCompletion,
};
class CompletionLookup {
    // eslint-disable-next-line no-useless-constructor
    constructor(cmdId, name, description) {
        this.cmdId = cmdId;
        this.name = name;
        this.description = description;
        this.blocklistMap = {
            app: ['apps:create'],
            space: ['spaces:create'],
        };
        this.keyAliasMap = {
            key: {
                'config:get': 'config',
            },
        };
        this.commandArgsMap = {
            key: {
                'config:set': 'configSet',
            },
        };
    }
    get key() {
        return this.argAlias() || this.keyAlias() || this.descriptionAlias() || this.name;
    }
    run() {
        if (this.blocklisted())
            return;
        return exports.CompletionMapping[this.key];
    }
    argAlias() {
        return this.commandArgsMap[this.name] && this.commandArgsMap[this.name][this.cmdId];
    }
    keyAlias() {
        return this.keyAliasMap[this.name] && this.keyAliasMap[this.name][this.cmdId];
    }
    descriptionAlias() {
        const d = this.description;
        // eslint-disable-next-line unicorn/prefer-starts-ends-with
        if (d.match(/^dyno size/))
            return 'dynosize';
        // eslint-disable-next-line unicorn/prefer-starts-ends-with
        if (d.match(/^process type/))
            return 'processtype';
    }
    blocklisted() {
        return this.blocklistMap[this.name] && this.blocklistMap[this.name].includes(this.cmdId);
    }
}
exports.CompletionLookup = CompletionLookup;
