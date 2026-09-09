import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { assetBase, appName, gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <span className="flex items-center gap-2 font-semibold">
          <img src={`${assetBase}/logo.png`} alt="" className="h-7 w-auto" />
          {appName}
        </span>
      ),
    },
    links: [
      {
        text: 'GitHub',
        url: `https://github.com/${gitConfig.repo}`,
        external: true,
      },
    ],
  };
}