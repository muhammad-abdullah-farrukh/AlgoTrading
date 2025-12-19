import { useLocation, Link } from 'react-router-dom';
import {
  LayoutDashboard,
  History,
  Cog,
  Bot,
  Link as LinkIcon,
  Brain,
  Globe,
  TrendingUp,
} from 'lucide-react';
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarFooter,
  useSidebar,
} from '@/components/ui/sidebar';
import { cn } from '@/lib/utils';

const mainNavItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/connection', label: 'MT5 Connection', icon: LinkIcon },
  { path: '/history', label: 'Trade History', icon: History },
];

const tradingNavItems = [
  { path: '/trading', label: 'Trading', icon: TrendingUp },
  { path: '/strategies', label: 'Strategies', icon: Cog },
  { path: '/autotrading', label: 'Autotrading', icon: Bot },
];

const mlNavItems = [
  { path: '/models', label: 'Models', icon: Brain },
  { path: '/scraper', label: 'Data Pipeline', icon: Globe },
];

export function AppSidebar() {
  const location = useLocation();
  const { state } = useSidebar();
  const isCollapsed = state === 'collapsed';

  const isActive = (path: string) => location.pathname === path;

  const NavItem = ({ item }: { item: { path: string; label: string; icon: any } }) => (
    <SidebarMenuItem>
      <SidebarMenuButton
        asChild
        isActive={isActive(item.path)}
        tooltip={item.label}
        className="transition-all duration-200 hover:translate-x-1"
      >
        <Link to={item.path} className="flex items-center gap-3">
          <item.icon className={cn(
            "h-4 w-4 shrink-0 transition-colors duration-200",
            isActive(item.path) && "text-primary"
          )} />
          <span className={cn(
            "transition-all duration-200",
            isCollapsed && "opacity-0 w-0 overflow-hidden"
          )}>
            {item.label}
          </span>
        </Link>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="border-b border-border p-4">
        <Link to="/" className="flex items-center gap-2 group">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary transition-all duration-300 group-hover:shadow-lg group-hover:shadow-primary/30">
            <TrendingUp className="h-4 w-4 text-primary-foreground" />
          </div>
          <span className={cn(
            "font-bold text-lg text-foreground transition-all duration-200",
            isCollapsed && "opacity-0 w-0 overflow-hidden"
          )}>
            TradeBot
          </span>
        </Link>
      </SidebarHeader>

      <SidebarContent>
        {/* Main Navigation */}
        <SidebarGroup>
          <SidebarGroupLabel className={cn(isCollapsed && "sr-only")}>
            Overview
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {mainNavItems.map((item) => (
                <NavItem key={item.path} item={item} />
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Trading Navigation */}
        <SidebarGroup>
          <SidebarGroupLabel className={cn(isCollapsed && "sr-only")}>
            Trading
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {tradingNavItems.map((item) => (
                <NavItem key={item.path} item={item} />
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* ML & Data Navigation */}
        <SidebarGroup>
          <SidebarGroupLabel className={cn(isCollapsed && "sr-only")}>
            ML & Data
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {mlNavItems.map((item) => (
                <NavItem key={item.path} item={item} />
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-border p-4">
        <div className={cn(
          "text-xs text-muted-foreground transition-opacity duration-200",
          isCollapsed && "opacity-0"
        )}>
          Press <kbd className="px-1 py-0.5 bg-muted rounded text-xs">⌘B</kbd> to toggle
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
