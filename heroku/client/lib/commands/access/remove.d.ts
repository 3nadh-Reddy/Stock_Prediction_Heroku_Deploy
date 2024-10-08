import { Command } from '@heroku-cli/command';
export default class AccessRemove extends Command {
    static description: string;
    static example: string;
    static topic: string;
    static flags: {
        app: import("@oclif/core/lib/interfaces").OptionFlag<string, import("@oclif/core/lib/interfaces/parser").CustomOptions>;
        remote: import("@oclif/core/lib/interfaces").OptionFlag<string | undefined, import("@oclif/core/lib/interfaces/parser").CustomOptions>;
    };
    static strict: boolean;
    run(): Promise<void>;
}
